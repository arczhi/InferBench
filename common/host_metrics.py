from __future__ import annotations

import os
import shutil
import threading
import time
from typing import Any, Dict, Optional

from .util import JsonlWriter, utc_now


def _read(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except Exception:  # noqa: BLE001
        return None


def _meminfo() -> Dict[str, int]:
    out: Dict[str, int] = {}
    for line in (_read("/proc/meminfo") or "").splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        parts = rest.strip().split()
        if not parts:
            continue
        try:
            value = int(parts[0])
        except ValueError:
            continue
        if len(parts) > 1 and parts[1].lower() == "kb":
            value //= 1024
        out[key.strip()] = value
    return out


def _cpu_total_idle() -> Optional[tuple[int, int]]:
    for line in (_read("/proc/stat") or "").splitlines():
        if line.startswith("cpu "):
            parts = [int(x) for x in line.split()[1:]]
            if len(parts) < 4:
                return None
            idle = parts[3] + (parts[4] if len(parts) > 4 else 0)
            return sum(parts), idle
    return None


class HostSampler(threading.Thread):
    def __init__(self, out_path: str, disk_path: str, interval_s: float = 1.0) -> None:
        super().__init__(daemon=True, name="host-sampler")
        self.interval_s = interval_s
        self.disk_path = disk_path
        self.writer = JsonlWriter(out_path)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest: Dict[str, Any] = {}
        self._prev = _cpu_total_idle()
        self._tick = 0

    def _sample(self) -> Dict[str, Any]:
        mem = _meminfo()
        cur = _cpu_total_idle()
        cpu_pct = None
        if self._prev and cur:
            dt = cur[0] - self._prev[0]
            di = cur[1] - self._prev[1]
            if dt > 0:
                cpu_pct = round(100.0 * (1.0 - di / dt), 1)
        self._prev = cur
        try:
            load1, load5, load15 = os.getloadavg()
        except OSError:
            load1 = load5 = load15 = None
        sample: Dict[str, Any] = {
            "ts": utc_now(),
            "mono": time.monotonic(),
            "cpu_pct": cpu_pct,
            "load1": load1,
            "load5": load5,
            "load15": load15,
            "mem_total_mib": mem.get("MemTotal"),
            "mem_avail_mib": mem.get("MemAvailable"),
            "mem_free_mib": mem.get("MemFree"),
            "swap_total_mib": mem.get("SwapTotal"),
            "swap_free_mib": mem.get("SwapFree"),
            "dirty_mib": mem.get("Dirty"),
        }
        self._tick += 1
        if self._tick % 10 == 1:
            try:
                usage = shutil.disk_usage(self.disk_path)
                sample["disk_free_gb"] = round(usage.free / 2**30, 2)
                sample["disk_used_pct"] = round(100.0 * usage.used / usage.total, 1)
            except Exception:  # noqa: BLE001
                pass
        return sample

    def run(self) -> None:
        while not self._stop_event.is_set():
            started = time.monotonic()
            sample = self._sample()
            with self._lock:
                self._latest = sample
            self.writer.write(sample)
            elapsed = time.monotonic() - started
            self._stop_event.wait(max(0.1, self.interval_s - elapsed))

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._latest)

    def stop(self) -> None:
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=10)
        self.writer.close()
