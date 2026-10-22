from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

from .util import JsonlWriter, run_cmd, utc_now

QUERY_FIELDS = [
    "index",
    "name",
    "memory.total",
    "memory.used",
    "memory.free",
    "utilization.gpu",
    "utilization.memory",
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "clocks.sm",
    "clocks.mem",
    "pstate",
]


def _num(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if text.lower() in {"", "n/a", "[n/a]", "not supported", "[not supported]"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def query_gpu() -> Dict[str, Any]:
    res = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=" + ",".join(QUERY_FIELDS),
            "--format=csv,noheader,nounits",
        ],
        timeout=5,
    )
    sample: Dict[str, Any] = {"ts": utc_now(), "mono": time.monotonic(), "ok": res["ok"]}
    if not res["ok"]:
        sample["error"] = res["err"] or "nvidia-smi failed"
        return sample
    line = res["out"].splitlines()[0] if res["out"] else ""
    cols = [c.strip() for c in line.split(",")]
    if len(cols) != len(QUERY_FIELDS):
        sample["ok"] = False
        sample["error"] = f"unexpected nvidia-smi output: {line[:200]}"
        return sample
    row = dict(zip(QUERY_FIELDS, cols))
    total = _num(row.get("memory.total"))
    used = _num(row.get("memory.used"))
    free = _num(row.get("memory.free"))
    sample.update(
        {
            "index": int(_num(row.get("index")) or 0),
            "name": row.get("name"),
            "mem_total_mib": total,
            "mem_used_mib": used,
            "mem_free_mib": free,
            "mem_used_pct": round(100.0 * used / total, 1) if total and used is not None else None,
            "util_gpu_pct": _num(row.get("utilization.gpu")),
            "util_mem_pct": _num(row.get("utilization.memory")),
            "temp_c": _num(row.get("temperature.gpu")),
            "power_w": _num(row.get("power.draw")),
            "power_limit_w": _num(row.get("power.limit")),
            "sm_clock_mhz": _num(row.get("clocks.sm")),
            "mem_clock_mhz": _num(row.get("clocks.mem")),
            "pstate": row.get("pstate"),
        }
    )
    return sample


class GpuSampler(threading.Thread):
    def __init__(self, out_path: str, interval_s: float = 1.0) -> None:
        super().__init__(daemon=True, name="gpu-sampler")
        self.interval_s = interval_s
        self.writer = JsonlWriter(out_path)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest: Dict[str, Any] = {}
        self._history: list[Dict[str, Any]] = []

    def run(self) -> None:
        while not self._stop_event.is_set():
            started = time.monotonic()
            sample = query_gpu()
            with self._lock:
                self._latest = sample
                self._history.append(sample)
                if len(self._history) > 600:
                    del self._history[:100]
            self.writer.write(sample)
            elapsed = time.monotonic() - started
            self._stop_event.wait(max(0.1, self.interval_s - elapsed))

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._latest)

    def recent(self, n: int = 5) -> list[Dict[str, Any]]:
        with self._lock:
            return list(self._history[-n:])

    def stop(self) -> None:
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=10)
        self.writer.close()
