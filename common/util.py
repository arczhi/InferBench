from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Optional, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(ROOT, "runs")


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")


def local_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def atomic_write_json(path: str, payload: Any) -> None:
    tmp = path + ".tmp"
    ensure_dir(os.path.dirname(path) or ".")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


class JsonlWriter:
    """Thread-safe append-only JSONL writer. Flushes every record."""

    def __init__(self, path: str) -> None:
        ensure_dir(os.path.dirname(path) or ".")
        self.path = path
        self._lock = threading.Lock()
        self._fh = open(path, "a", encoding="utf-8")

    def write(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.close()
            except Exception:  # noqa: BLE001
                pass


def run_cmd(
    cmd: Sequence[str],
    timeout: float = 30.0,
    check: bool = False,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    started = time.time()
    try:
        proc = subprocess.run(
            list(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True,
            errors="replace",
            stdin=subprocess.DEVNULL,
            env=env,
        )
        result = {
            "ok": proc.returncode == 0,
            "rc": proc.returncode,
            "out": proc.stdout.strip(),
            "err": proc.stderr.strip(),
            "ms": round(1000 * (time.time() - started), 1),
            "cmd": list(cmd),
        }
    except FileNotFoundError:
        result = {"ok": False, "rc": None, "out": "", "err": "not found", "ms": 0, "cmd": list(cmd)}
    except subprocess.TimeoutExpired:
        result = {"ok": False, "rc": None, "out": "", "err": "timeout", "ms": round(1000 * (time.time() - started), 1), "cmd": list(cmd)}
    except Exception as exc:  # noqa: BLE001
        result = {"ok": False, "rc": None, "out": "", "err": repr(exc), "ms": 0, "cmd": list(cmd)}
    if check and not result["ok"]:
        raise RuntimeError(f"command failed: {cmd}\n{result['err']}")
    return result


def which(name: str) -> Optional[str]:
    from shutil import which as _which

    return _which(name)


def human_gib(mib: Optional[float]) -> str:
    if mib is None:
        return "?"
    return f"{mib / 1024:.1f}GiB"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def fail(msg: str, code: int = 2) -> None:
    print(f"[FAIL] {msg}", file=sys.stderr)
    raise SystemExit(code)
