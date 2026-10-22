from __future__ import annotations

import collections
import os
import threading
import time
from typing import Any, Callable, Deque, Dict, Optional

from .config import Config
from .gpu_metrics import GpuSampler
from .host_metrics import HostSampler
from .util import JsonlWriter, atomic_write_json, utc_now


class Guard(threading.Thread):
    """Independent watchdog. Trips on thermal, memory, disk, load, OOM, wall time.

    The guard never kills processes directly; it sets `abort_event` and the
    runner stops the experiment container in its `finally` path.
    """

    def __init__(
        self,
        cfg: Config,
        run_dir: str,
        gpu: GpuSampler,
        host: HostSampler,
        engine_state: Callable[[], Dict[str, Any]],
        run_started_mono: float,
    ) -> None:
        super().__init__(daemon=True, name="safety-guard")
        self.cfg = cfg
        self.thresholds = cfg.guard
        self.run_dir = run_dir
        self.gpu = gpu
        self.host = host
        self.engine_state = engine_state
        self.run_started = run_started_mono
        self.abort_event = threading.Event()
        self.reason: Optional[str] = None
        self.detail: Optional[Dict[str, Any]] = None
        self._stop_event = threading.Event()
        self._writer = JsonlWriter(os.path.join(run_dir, "guard.jsonl"))
        self._errors: Deque[bool] = collections.deque(maxlen=self.thresholds.error_window)
        self._errors_lock = threading.Lock()
        self._counters: Dict[str, int] = collections.defaultdict(int)
        self._last_warn: Dict[str, float] = {}
        self._tick = 0
        self._last_engine_check = 0.0
        self._engine_cached: Dict[str, Any] = {}
        self.ncpu = os.cpu_count() or 4

    def note_request(self, ok: bool) -> None:
        with self._errors_lock:
            self._errors.append(bool(ok))

    def _error_rate(self) -> Optional[float]:
        with self._errors_lock:
            if len(self._errors) < max(5, self.thresholds.error_window // 2):
                return None
            return sum(1 for ok in self._errors if not ok) / len(self._errors)

    def trip(self, reason: str, detail: Dict[str, Any]) -> None:
        if self.abort_event.is_set():
            return
        self.reason = reason
        self.detail = detail
        record = {
            "ts": utc_now(),
            "event": "trip",
            "reason": reason,
            "detail": detail,
            "elapsed_s": round(time.monotonic() - self.run_started, 1),
        }
        self._writer.write(record)
        atomic_write_json(os.path.join(self.run_dir, "abort.json"), record)
        print(f"\n[GUARD] ABORT: {reason} :: {detail}")
        self.abort_event.set()

    def _warn(self, key: str, message: str, detail: Dict[str, Any], every_s: float = 60.0) -> None:
        now = time.monotonic()
        if now - self._last_warn.get(key, 0.0) < every_s:
            return
        self._last_warn[key] = now
        self._writer.write(
            {
                "ts": utc_now(),
                "event": "warn",
                "key": key,
                "message": message,
                "detail": detail,
                "elapsed_s": round(now - self.run_started, 1),
            }
        )
        print(f"[GUARD] WARN: {message}")

    def _consecutive(self, key: str, condition: bool, needed: int) -> bool:
        if condition:
            self._counters[key] += 1
        else:
            self._counters[key] = 0
        return self._counters[key] >= needed

    def run(self) -> None:
        while not self._stop_event.is_set():
            started = time.monotonic()
            try:
                self._check()
            except Exception as exc:  # noqa: BLE001
                self._warn("guard-error", f"guard check failed: {exc}", {}, every_s=30)
            elapsed = time.monotonic() - started
            self._stop_event.wait(max(0.2, self.thresholds.sample_interval_s - elapsed))

    def _check(self) -> None:
        t = self.thresholds
        self._tick += 1
        gpu = self.gpu.snapshot()
        host = self.host.snapshot()
        now = time.monotonic()
        elapsed = now - self.run_started

        if self.abort_event.is_set():
            return

        if elapsed > t.max_wall_s:
            self.trip("wall_time_exceeded", {"elapsed_s": round(elapsed, 1), "max_wall_s": t.max_wall_s})
            return

        temp = gpu.get("temp_c")
        if temp is not None:
            hot = temp >= t.gpu_temp_abort_c
            if self._consecutive("gpu_temp", hot, 2):
                self.trip("gpu_temperature", {"temp_c": temp, "limit_c": t.gpu_temp_abort_c})
                return
            if temp >= t.gpu_temp_warn_c:
                self._warn("gpu-temp", f"GPU temperature {temp}C", {"temp_c": temp})
        else:
            self._counters["gpu_temp"] = 0

        used_pct = gpu.get("mem_used_pct")
        if used_pct is not None:
            high = used_pct >= t.gpu_mem_used_abort_pct
            if self._consecutive("gpu_mem", high, 2):
                self.trip(
                    "gpu_memory_exhaustion",
                    {
                        "mem_used_pct": used_pct,
                        "mem_used_mib": gpu.get("mem_used_mib"),
                        "mem_total_mib": gpu.get("mem_total_mib"),
                    },
                )
                return
        else:
            self._counters["gpu_mem"] = 0

        avail = host.get("mem_avail_mib")
        if avail is not None:
            low = avail <= t.host_mem_avail_abort_mib
            if self._consecutive("host_mem", low, 3):
                self.trip("host_memory_low", {"mem_avail_mib": avail, "limit_mib": t.host_mem_avail_abort_mib})
                return
            if avail <= t.host_mem_avail_abort_mib * 2:
                self._warn("host-mem", f"host memory available {avail} MiB", {"mem_avail_mib": avail})
        else:
            self._counters["host_mem"] = 0

        disk_free = host.get("disk_free_gb")
        if disk_free is not None and disk_free <= t.disk_free_abort_gb:
            self.trip("disk_low", {"disk_free_gb": disk_free, "limit_gb": t.disk_free_abort_gb})
            return

        load1 = host.get("load1")
        if load1 is not None:
            high = load1 > self.ncpu * t.load_per_core_abort
            if self._consecutive("load", high, 60):
                self.trip("host_load_high", {"load1": load1, "ncpu": self.ncpu})
                return
            if load1 > self.ncpu * 1.2:
                self._warn("load", f"host load {load1} with {self.ncpu} cores", {"load1": load1})
        else:
            self._counters["load"] = 0

        err_rate = self._error_rate()
        if err_rate is not None and err_rate >= t.error_rate_abort:
            self.trip("request_error_rate", {"error_rate": round(err_rate, 3), "window": len(self._errors)})
            return

        if now - self._last_engine_check >= 5.0:
            self._last_engine_check = now
            try:
                self._engine_cached = self.engine_state() or {}
            except Exception as exc:  # noqa: BLE001
                self._engine_cached = {"status": "unknown", "error": str(exc)}
            state = self._engine_cached
            if state.get("oom_killed"):
                self.trip("container_oom_killed", state)
                return
            if state.get("status") in ("exited", "dead"):
                self.trip("container_exited", state)
                return

        if self._tick % 30 == 0:
            self._writer.write(
                {
                    "ts": utc_now(),
                    "event": "heartbeat",
                    "elapsed_s": round(elapsed, 1),
                    "gpu_temp_c": temp,
                    "gpu_mem_used_pct": used_pct,
                    "host_mem_avail_mib": avail,
                    "host_load1": load1,
                    "error_rate": err_rate,
                    "engine": self._engine_cached,
                }
            )

    def stop(self) -> None:
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=10)
        self._writer.close()
