#!/usr/bin/env python3
"""Read-only survey of an L4 inference host.

Stdlib only. Does not modify the machine, does not pull images, does not
start containers. Safe to run while other services are live.

Usage:
    python3 survey_l4.py
    python3 survey_l4.py --no-network
    python3 survey_l4.py --json-out /tmp/survey.json --quiet
    python3 survey_l4.py --docker-gpu-probe      # uses a LOCAL image only

The JSON file is the machine fact sheet consumed by the experiment config.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import re
import shutil
import socket
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

SCHEMA = "inference-exp.survey/v1"
COMMON_PORTS = [8000, 30000, 30001, 8080, 8888]
KV_TABLE = {
    # bytes per token for fp16 KV, (layers, kv_heads, head_dim)
    "qwen3-4b": (36, 8, 128),
    "qwen3-8b": (36, 8, 128),
    "qwen2.5-7b": (28, 4, 128),
}
WEIGHT_GB = {"qwen3-4b": 8.5, "qwen3-8b": 16.5, "qwen2.5-7b": 15.5}


def _run(cmd: List[str], timeout: float = 8.0, stdin_null: bool = True) -> Dict[str, Any]:
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True,
            errors="replace",
            stdin=subprocess.DEVNULL if stdin_null else None,
        )
        return {
            "ok": p.returncode == 0,
            "rc": p.returncode,
            "out": p.stdout.strip(),
            "err": p.stderr.strip(),
        }
    except FileNotFoundError:
        return {"ok": False, "rc": None, "out": "", "err": "not found"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "rc": None, "out": "", "err": "timeout"}
    except Exception as exc:  # noqa: BLE001 - survey must never crash
        return {"ok": False, "rc": None, "out": "", "err": repr(exc)}


def _read(path: str, limit: int = 200_000) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read(limit)
    except Exception:  # noqa: BLE001
        return None


def _f(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if text.lower() in {"", "n/a", "[n/a]", "not supported", "[not supported]", "unknown"}:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _i(value: Any) -> Optional[int]:
    num = _f(value)
    return int(num) if num is not None else None


def _gib(mib: Optional[float]) -> Optional[float]:
    return round(mib / 1024.0, 2) if mib is not None else None


def _meminfo() -> Dict[str, int]:
    text = _read("/proc/meminfo") or ""
    out: Dict[str, int] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        parts = rest.strip().split()
        if not parts:
            continue
        try:
            val = int(parts[0])
        except ValueError:
            continue
        if len(parts) > 1 and parts[1].lower() == "kb":
            val = int(val / 1024)
        out[key.strip()] = val
    if not out and platform.system() == "Darwin":
        total = _run(["sysctl", "-n", "hw.memsize"], timeout=3)
        page = _run(["sysctl", "-n", "hw.pagesize"], timeout=3)
        try:
            total_mib = int(total["out"]) // 2**20
            out["MemTotal"] = total_mib
            out["MemAvailable"] = total_mib
            if page["ok"]:
                vm = _run(["vm_stat"], timeout=3)
                if vm["ok"]:
                    free_pages = 0
                    wanted = ("pages free", "pages inactive", "pages speculative", "pages purgeable")
                    for line in vm["out"].splitlines():
                        lowered = line.lower()
                        if any(lowered.startswith(key) for key in wanted):
                            free_pages += int(re.sub(r"[^0-9]", "", line.split(":")[-1]) or 0)
                    if free_pages:
                        out["MemAvailable"] = int(free_pages * int(page["out"]) / 2**20)
        except (ValueError, TypeError):
            pass
    return out


def _cpu_stat_sample() -> Optional[Dict[str, int]]:
    text = _read("/proc/stat")
    if not text:
        return None
    for line in text.splitlines():
        if line.startswith("cpu "):
            parts = [int(x) for x in line.split()[1:]]
            if len(parts) < 4:
                return None
            idle = parts[3] + (parts[4] if len(parts) > 4 else 0)
            return {"total": sum(parts), "idle": idle}
    return None


def _cpu_percent(window: float = 0.4) -> Optional[float]:
    first = _cpu_stat_sample()
    if not first:
        return None
    time.sleep(window)
    second = _cpu_stat_sample()
    if not second:
        return None
    dt = second["total"] - first["total"]
    di = second["idle"] - first["idle"]
    if dt <= 0:
        return None
    return round(100.0 * (1.0 - di / dt), 1)


def _loadavg() -> Optional[List[float]]:
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except OSError:
        return None


def _distro() -> Dict[str, str]:
    text = _read("/etc/os-release") or ""
    data: Dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            data[k.strip()] = v.strip().strip('"')
    return {
        "id": data.get("ID", ""),
        "version": data.get("VERSION_ID", ""),
        "pretty": data.get("PRETTY_NAME", ""),
    }


def _uptime_s() -> Optional[float]:
    text = _read("/proc/uptime")
    if not text:
        return None
    try:
        return float(text.split()[0])
    except (ValueError, IndexError):
        return None


def _shm() -> Optional[Dict[str, Any]]:
    try:
        usage = shutil.disk_usage("/dev/shm")
        return {"total_mib": round(usage.total / 2**20), "free_mib": round(usage.free / 2**20)}
    except Exception:  # noqa: BLE001
        return None


def _no_file_limit() -> Optional[int]:
    try:
        import resource

        soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        return int(soft)
    except Exception:  # noqa: BLE001
        return None


def _disk(path: str) -> Optional[Dict[str, Any]]:
    try:
        usage = shutil.disk_usage(path)
        return {
            "path": path,
            "total_gb": round(usage.total / 2**30, 2),
            "free_gb": round(usage.free / 2**30, 2),
            "used_pct": round(100.0 * usage.used / usage.total, 1) if usage.total else None,
        }
    except Exception:  # noqa: BLE001
        return None


def _ports_listening() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    ss = _run(["ss", "-ltnH"], timeout=5)
    if ss["ok"] and ss["out"]:
        ports = []
        for line in ss["out"].splitlines():
            cols = line.split()
            if len(cols) >= 4:
                local = cols[3]
                port = local.rsplit(":", 1)[-1]
                if port.isdigit():
                    ports.append(int(port))
        out["via"] = "ss"
        out["ports"] = sorted(set(ports))
    else:
        ports = []
        proc_net = _read("/proc/net/tcp") or ""
        for line in proc_net.splitlines()[1:]:
            parts = line.split()
            if len(parts) > 1:
                try:
                    ports.append(int(parts[1].rsplit(":", 1)[1], 16))
                except (ValueError, IndexError):
                    pass
        out["via"] = "proc"
        out["ports"] = sorted(set(ports))
    out["listen_common"] = {str(p): p in out["ports"] for p in COMMON_PORTS}
    return out


def _docker_json(args: List[str], timeout: float = 10.0) -> Any:
    res = _run(["docker"] + args, timeout=timeout)
    if not res["ok"] or not res["out"]:
        return None
    try:
        return json.loads(res["out"])
    except json.JSONDecodeError:
        return None


def _docker() -> Dict[str, Any]:
    data: Dict[str, Any] = {"installed": shutil.which("docker") is not None}
    if not data["installed"]:
        return data
    version = _docker_json(["version", "--format", "{{json .}}"])
    info = _docker_json(["info", "--format", "{{json .}}"], timeout=15)
    data["version"] = version
    data["info"] = info
    if info:
        data["root_dir"] = info.get("DockerRootDir")
        data["runtimes"] = info.get("Runtimes")
        data["storage_driver"] = info.get("Driver")
        data["cgroup_driver"] = info.get("CgroupDriver")
        data["cgroup_version"] = info.get("CgroupVersion")
        data["server_version"] = info.get("ServerVersion")
        data["nvidia_runtime"] = "nvidia" in (info.get("Runtimes") or {})
        data["daemon_labels"] = info.get("Labels")
        reg = info.get("RegistryConfig") or {}
        data["registry_mirrors"] = reg.get("Mirrors")
        data["insecure_registries"] = reg.get("InsecureRegistryCIDRs")
    images: List[Dict[str, Any]] = []
    res = _run(
        [
            "docker",
            "images",
            "--format",
            "{{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.Size}}\t{{.CreatedSince}}",
        ],
        timeout=15,
    )
    if res["ok"]:
        for line in res["out"].splitlines():
            cols = line.split("\t")
            if len(cols) >= 5:
                images.append(
                    {
                        "repository": cols[0],
                        "tag": cols[1],
                        "id": cols[2],
                        "size": cols[3],
                        "created": cols[4],
                        "relevant": any(k in cols[0].lower() for k in ("vllm", "sglang")),
                    }
                )
    data["images"] = images
    containers: List[Dict[str, Any]] = []
    res = _run(
        [
            "docker",
            "ps",
            "-a",
            "--format",
            "{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}\t{{.Label \"inference-exp\"}}",
        ],
        timeout=15,
    )
    if res["ok"]:
        for line in res["out"].splitlines():
            cols = line.split("\t")
            if len(cols) >= 4:
                containers.append(
                    {
                        "name": cols[0],
                        "image": cols[1],
                        "status": cols[2],
                        "ports": cols[3],
                        "inference_exp": (cols[4].strip() if len(cols) > 4 else "") == "1",
                    }
                )
    data["containers"] = containers
    volumes = _run(["docker", "volume", "ls", "--format", "{{.Name}}"], timeout=10)
    data["volumes"] = volumes["out"].splitlines() if volumes["ok"] else []
    data["daemon_json"] = None
    text = _read("/etc/docker/daemon.json", 100_000)
    if text:
        try:
            data["daemon_json"] = json.loads(text)
        except json.JSONDecodeError:
            data["daemon_json"] = {"_unparsed": text[:2000]}
    return data


def _nvidia_query(fields: List[str], suffix: str = "gpu") -> Optional[List[Dict[str, Optional[float]]]]:
    cmd = [
        "nvidia-smi",
        "--query-" + suffix + "=" + ",".join(fields),
        "--format=csv,noheader,nounits",
    ]
    res = _run(cmd, timeout=10)
    if not res["ok"] or not res["out"]:
        return None
    rows: List[Dict[str, Optional[float]]] = []
    for line in res["out"].splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) != len(fields):
            continue
        rows.append(dict(zip(fields, cols)))
    return rows


def _gpu() -> Dict[str, Any]:
    data: Dict[str, Any] = {"nvidia_smi": shutil.which("nvidia-smi") is not None}
    if not data["nvidia_smi"]:
        lspci = _run(["bash", "-lc", "lspci 2>/dev/null | grep -i nvidia || true"], timeout=5)
        data["lspci"] = lspci["out"]
        return data

    header = _run(["nvidia-smi"], timeout=10)
    data["header"] = header["out"][:4000] if header["ok"] else ""
    cuda = None
    m = re.search(r"CUDA Version:\s*([0-9.]+)", data["header"] or "")
    if m:
        cuda = m.group(1)
    data["cuda_version"] = cuda
    driver = None
    m = re.search(r"Driver Version:\s*([0-9.]+)", data["header"] or "")
    if m:
        driver = m.group(1)
    data["driver_version"] = driver

    core_fields = [
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
    extra_fields = [
        "index",
        "compute_cap",
        "power.default_limit",
        "power.min_limit",
        "power.max_limit",
        "pcie.link.gen.current",
        "pcie.link.width.current",
    ]
    rows = _nvidia_query(core_fields)
    if rows is None:
        rows = _nvidia_query(
            ["index", "name", "memory.total", "memory.used", "memory.free", "temperature.gpu", "utilization.gpu"]
        )
    extras = _nvidia_query(extra_fields) or []
    extras_by_index = {str(r.get("index")): r for r in extras}
    for row in rows or []:
        extra_row = extras_by_index.get(str(row.get("index")))
        if extra_row:
            for key, value in extra_row.items():
                if key != "index":
                    row[key] = value
    gpus: List[Dict[str, Any]] = []
    for row in rows or []:
        gpu = {
            "index": _i(row.get("index")),
            "name": row.get("name"),
            "compute_cap": _f(row.get("compute_cap")),
            "memory_total_mib": _i(row.get("memory.total")),
            "memory_used_mib": _i(row.get("memory.used")),
            "memory_free_mib": _i(row.get("memory.free")),
            "utilization_gpu_pct": _f(row.get("utilization.gpu")),
            "utilization_mem_pct": _f(row.get("utilization.memory")),
            "temperature_c": _f(row.get("temperature.gpu")),
            "power_w": _f(row.get("power.draw")),
            "power_limit_w": _f(row.get("power.limit")),
            "power_default_limit_w": _f(row.get("power.default_limit")),
            "power_min_limit_w": _f(row.get("power.min_limit")),
            "power_max_limit_w": _f(row.get("power.max_limit")),
            "sm_clock_mhz": _f(row.get("clocks.sm")),
            "mem_clock_mhz": _f(row.get("clocks.mem")),
            "pstate": row.get("pstate"),
            "pcie_gen": _i(row.get("pcie.link.gen.current")),
            "pcie_width": _i(row.get("pcie.link.width.current")),
        }
        if gpu["memory_total_mib"]:
            gpu["memory_used_pct"] = round(
                100.0 * (gpu["memory_used_mib"] or 0) / gpu["memory_total_mib"], 1
            )
        gpus.append(gpu)
    data["gpus"] = gpus

    extra_fields = [
        "ecc.mode.current",
        "mig.mode.current",
        "persistence_mode",
        "compute_mode",
        "serial",
        "vbios_version",
    ]
    extra_rows = _nvidia_query(extra_fields)
    extra: Dict[str, Any] = {}
    for row in extra_rows or []:
        extra = row
        break
    data["features"] = extra

    apps = _nvidia_query(["pid", "process_name", "used_memory"], suffix="compute-apps")
    data["compute_apps"] = [
        {
            "pid": _i(r.get("pid")),
            "process": r.get("process_name"),
            "used_memory_mib": _i(r.get("used_memory")),
        }
        for r in (apps or [])
    ]
    data["nvidia_container_cli"] = _run(["nvidia-container-cli", "--version"], timeout=5)["out"] or None
    data["nvidia_ctk"] = _run(["nvidia-ctk", "--version"], timeout=5)["out"] or None
    data["proc_driver"] = (_read("/proc/driver/nvidia/version") or "")[:500] or None
    data["nvidia_persistenced"] = _run(
        ["bash", "-lc", "pgrep -a nvidia-persistenced 2>/dev/null || true"], timeout=5
    )["out"] or None
    return data


def _probe(url: str, timeout: float = 6.0) -> Dict[str, Any]:
    started = time.time()
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": "inference-exp-survey/1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return {
                "ok": True,
                "status": resp.status,
                "ms": round(1000 * (time.time() - started), 1),
                "final_url": resp.geturl(),
            }
    except urllib.error.HTTPError as exc:
        return {
            "ok": exc.code in (200, 204, 401, 403),
            "status": exc.code,
            "ms": round(1000 * (time.time() - started), 1),
            "error": "auth-required" if exc.code in (401, 403) else exc.reason,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "status": None,
            "ms": round(1000 * (time.time() - started), 1),
            "error": str(exc)[:240],
        }


def _dns(host: str) -> Dict[str, Any]:
    started = time.time()
    try:
        infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        addrs = sorted({info[4][0] for info in infos})
        return {"ok": True, "addrs": addrs[:4], "ms": round(1000 * (time.time() - started), 1)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)[:200], "ms": round(1000 * (time.time() - started), 1)}


def _network(model: str) -> Dict[str, Any]:
    endpoints = {
        "docker_mirror": "https://docker.m.daocloud.io/v2/",
        "docker_hub": "https://registry-1.docker.io/v2/",
        "huggingface": "https://huggingface.co/api/models/" + model,
        "hf_mirror": "https://hf-mirror.com/api/models/" + model,
    }
    data: Dict[str, Any] = {}
    for name, url in endpoints.items():
        host = url.split("/")[2]
        data[name] = {"dns": _dns(host), "https": _probe(url)}
    return data


def _hf_cache() -> Dict[str, Any]:
    candidates = [
        os.path.expanduser("~/.cache/huggingface/hub"),
        "/root/.cache/huggingface/hub",
        os.path.expanduser("~/hf-cache/hub"),
    ]
    out: Dict[str, Any] = {"candidates": [], "models": []}
    for path in candidates:
        if os.path.isdir(path):
            entry: Dict[str, Any] = {"path": path}
            try:
                total = 0
                models = []
                for name in os.listdir(path):
                    if not name.startswith("models--"):
                        continue
                    full = os.path.join(path, name)
                    size = 0
                    for root, _dirs, files in os.walk(full):
                        for fname in files:
                            try:
                                size += os.path.getsize(os.path.join(root, fname))
                            except OSError:
                                pass
                    total += size
                    models.append({"name": name.replace("models--", "").replace("--", "/"), "gb": round(size / 2**30, 2)})
                entry["total_gb"] = round(total / 2**30, 2)
                entry["models"] = models
                out["models"].extend(models)
            except Exception as exc:  # noqa: BLE001
                entry["error"] = str(exc)[:200]
            out["candidates"].append(entry)
    return out


def _recommend(facts: Dict[str, Any]) -> Dict[str, Any]:
    gpus = facts.get("gpu", {}).get("gpus") or []
    mem = facts.get("memory", {})
    docker = facts.get("docker", {})
    osinfo = facts.get("os", {})
    ncpu = facts.get("cpu", {}).get("logical") or 4
    total_mib = gpus[0].get("memory_total_mib") if gpus else None
    used_mib = gpus[0].get("memory_used_mib") if gpus else None
    foreign_mib = sum(
        app.get("used_memory_mib") or 0
        for app in facts.get("gpu", {}).get("compute_apps", [])
    )
    ram_gb = (mem.get("total_mib") or 0) / 1024.0
    free_gb = (mem.get("available_mib") or 0) / 1024.0

    gpu_util = 0.9
    if foreign_mib > 4096:
        gpu_util = 0.8
    elif foreign_mib > 512:
        gpu_util = 0.85

    container_gb = int(max(8, min(28, ram_gb * 0.45)))
    cpus = int(max(2, min(8, ncpu // 2)))
    kv_mib = None
    kv_tokens_fp16 = None
    kv_tokens_fp8 = None
    if total_mib:
        weights_gb = WEIGHT_GB["qwen3-4b"]
        kv_gb = gpu_util * total_mib / 1024.0 - weights_gb - 1.5
        if kv_gb > 0:
            layers, heads, dim = KV_TABLE["qwen3-4b"]
            bytes_fp16 = 2 * layers * heads * dim * 2
            kv_mib = round(kv_gb * 1024, 1)
            kv_tokens_fp16 = int(kv_gb * 2**30 / bytes_fp16)
            kv_tokens_fp8 = int(kv_gb * 2**30 / (bytes_fp16 / 2))

    issues: List[Dict[str, str]] = []
    if not gpus:
        issues.append({"level": "fail", "msg": "nvidia-smi / GPU not visible"})
    else:
        name = gpus[0].get("name") or ""
        cap = gpus[0].get("compute_cap")
        if "L4" not in name.upper():
            issues.append({"level": "warn", "msg": f"GPU is {name}, not L4"})
        if cap is not None and abs(cap - 8.9) > 0.01:
            issues.append({"level": "warn", "msg": f"compute capability {cap}, expected 8.9 (sm89)"})
        if used_mib and used_mib > 512:
            issues.append(
                {
                    "level": "warn",
                    "msg": f"GPU already using {used_mib} MiB; 0.9 watermark may not fit",
                }
            )
        temp = gpus[0].get("temperature_c")
        if temp and temp > 60:
            issues.append({"level": "warn", "msg": f"GPU idle temperature {temp}C"})
    if not docker.get("installed"):
        issues.append({"level": "fail", "msg": "docker CLI not found"})
    else:
        if not docker.get("nvidia_runtime"):
            issues.append(
                {
                    "level": "warn",
                    "msg": "docker Runtimes has no 'nvidia' entry; verify with --docker-gpu-probe",
                }
            )
    if ram_gb and ram_gb < 16:
        issues.append({"level": "warn", "msg": f"host RAM only {ram_gb:.1f} GiB"})
    if free_gb and free_gb < 6:
        issues.append({"level": "fail", "msg": f"only {free_gb:.1f} GiB RAM available now"})
    elif free_gb and free_gb < 12:
        issues.append({"level": "warn", "msg": f"only {free_gb:.1f} GiB RAM available now"})
    swap_mib = mem.get("swap_total_mib") or 0
    if swap_mib > 0 and (mem.get("swap_free_mib") or 0) < swap_mib * 0.2:
        issues.append({"level": "warn", "msg": "swap mostly used"})
    root_disk = facts.get("disk", {}).get("docker_root") or facts.get("disk", {}).get("root")
    if root_disk and root_disk.get("free_gb") is not None:
        if root_disk["free_gb"] < 20:
            issues.append({"level": "fail", "msg": f"docker root only {root_disk['free_gb']} GiB free"})
        elif root_disk["free_gb"] < 50:
            issues.append({"level": "warn", "msg": f"docker root {root_disk['free_gb']} GiB free; vLLM+SGLang need ~30-40 GiB"})
    if shutil.which("python3") is None:
        issues.append({"level": "fail", "msg": "python3 not found"})
    if (facts.get("uptime_s") or 0) and (facts.get("uptime_s") or 0) < 300:
        issues.append({"level": "warn", "msg": "host booted <5 min ago"})
    if osinfo.get("arch") not in ("x86_64", "amd64"):
        issues.append({"level": "warn", "msg": f"arch {osinfo.get('arch')} (expect x86_64)"})
    if facts.get("loadavg") and ncpu and facts["loadavg"][0] > ncpu * 0.8:
        issues.append({"level": "warn", "msg": "host load is already high"})

    net = facts.get("network") or {}
    mirror = net.get("docker_mirror", {}).get("https", {})
    if net and not mirror.get("ok"):
        issues.append({"level": "warn", "msg": "docker.m.daocloud.io not reachable (or TLS issue)"})
    hf = net.get("hf_mirror", {}).get("https", {})
    hf_main = net.get("huggingface", {}).get("https", {})
    if net and not hf.get("ok") and not hf_main.get("ok"):
        issues.append({"level": "warn", "msg": "neither hf-mirror.com nor huggingface.co reachable; pre-fetch the model"})

    common = (facts.get("ports") or {}).get("listen_common") or {}
    busy = [p for p, is_busy in common.items() if is_busy]
    if busy:
        issues.append({"level": "warn", "msg": "ports already listening: " + ", ".join(busy)})

    return {
        "gpu_memory_utilization": gpu_util,
        "container_memory_gb": container_gb,
        "container_cpus": cpus,
        "cpu_shares": 512,
        "max_model_len": 32768,
        "estimated_kv_budget_mib_qwen3_4b": kv_mib,
        "estimated_kv_tokens_fp16_qwen3_4b": kv_tokens_fp16,
        "estimated_kv_tokens_fp8_qwen3_4b": kv_tokens_fp8,
        "suggested_ports": {"vllm": 8000, "sglang": 30000},
        "issues": issues,
    }


def _fmt_issue(issue: Dict[str, str]) -> str:
    tag = {"fail": "[FAIL]", "warn": "[WARN]", "ok": "[ OK ]"}.get(issue["level"], "[INFO]")
    return f"{tag} {issue['msg']}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Read-only L4 inference host survey")
    ap.add_argument("--json-out", default=None, help="write JSON facts here")
    ap.add_argument("--no-network", action="store_true", help="skip registry/HF reachability probes")
    ap.add_argument("--docker-gpu-probe", action="store_true", help="run nvidia-smi in a LOCAL image (no pull)")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--quiet", action="store_true", help="only print the summary block")
    args = ap.parse_args()

    started = time.time()
    cwd = os.getcwd()
    mem = _meminfo()
    mem_facts = {
        "total_mib": mem.get("MemTotal"),
        "available_mib": mem.get("MemAvailable"),
        "free_mib": mem.get("MemFree"),
        "cached_mib": mem.get("Cached"),
        "buffers_mib": mem.get("Buffers"),
        "swap_total_mib": mem.get("SwapTotal"),
        "swap_free_mib": mem.get("SwapFree"),
        "dirty_mib": mem.get("Dirty"),
        "overcommit": _read("/proc/sys/vm/overcommit_memory", 100),
        "thp": (_read("/proc/sys/kernel/mm/transparent_hugepage/enabled", 200) or "").strip(),
    }

    cpus = os.cpu_count() or 0
    cpuinfo = _read("/proc/cpuinfo") or ""
    model_name = None
    m = re.search(r"model name\s*:\s*(.+)", cpuinfo)
    if m:
        model_name = m.group(1).strip()
    if not model_name and platform.system() == "Darwin":
        model_name = _run(["sysctl", "-n", "machdep.cpu.brand_string"], timeout=3)["out"] or None
    physical = None
    lscpu = _run(["lscpu"], timeout=5)
    if lscpu["ok"]:
        m = re.search(r"^Core\(s\) per socket:\s*(\d+)", lscpu["out"], re.M)
        m2 = re.search(r"^Socket\(s\):\s*(\d+)", lscpu["out"], re.M)
        if m and m2:
            physical = int(m.group(1)) * int(m2.group(1))
        if not model_name:
            m3 = re.search(r"^Model name:\s*(.+)", lscpu["out"], re.M)
            if m3:
                model_name = m3.group(1).strip()

    docker = _docker()
    docker_root = docker.get("root_dir") or "/var/lib/docker"
    disk_facts = {
        "cwd": _disk(cwd),
        "root": _disk("/"),
        "docker_root": _disk(docker_root),
        "tmp": _disk("/tmp"),
    }
    gpu = _gpu()
    if args.docker_gpu_probe and docker.get("installed"):
        local_images = [
            img["repository"] + ":" + img["tag"]
            for img in docker.get("images", [])
            if img.get("repository") not in ("<none>", "none")
        ]
        probe_image = next((i for i in local_images if "vllm" in i or "sglang" in i), None)
        if probe_image:
            started_probe = time.time()
            res = _run(
                ["docker", "run", "--rm", "--gpus", "device=0", "--entrypoint", "nvidia-smi", probe_image],
                timeout=60,
            )
            gpu["docker_gpu_probe"] = {
                "image": probe_image,
                "ok": res["ok"],
                "ms": round(1000 * (time.time() - started_probe), 1),
                "out": res["out"][:1000],
                "err": res["err"][:1000],
            }
        else:
            gpu["docker_gpu_probe"] = {"ok": False, "error": "no local vllm/sglang image; skipped pull"}

    facts: Dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "duration_s": None,
        "host": {
            "hostname": socket.gethostname(),
            "cwd": cwd,
            "user": os.environ.get("USER") or os.environ.get("LOGNAME"),
        },
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "arch": platform.machine(),
            "distro": _distro(),
            "boot_time": _dt.datetime.fromtimestamp(
                time.time() - (_uptime_s() or 0)
            ).isoformat()
            if _uptime_s()
            else None,
        },
        "uptime_s": _uptime_s(),
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
            "venv": os.environ.get("VIRTUAL_ENV"),
        },
        "cpu": {
            "model": model_name,
            "logical": cpus,
            "physical": physical,
            "loadavg": _loadavg(),
            "utilization_pct": _cpu_percent(),
        },
        "memory": mem_facts,
        "disk": disk_facts,
        "shm": _shm(),
        "no_file_limit": _no_file_limit(),
        "ports": _ports_listening(),
        "docker": docker,
        "gpu": gpu,
        "hf_cache": _hf_cache(),
        "network": {} if args.no_network else _network(args.model),
        "loadavg": _loadavg(),
    }
    facts["recommendation"] = _recommend(facts)
    facts["duration_s"] = round(time.time() - started, 2)

    out_path = args.json_out
    if not out_path:
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = os.path.join(cwd, f"survey_{socket.gethostname()}_{stamp}.json")
    try:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(facts, fh, indent=2, ensure_ascii=False, default=str)
        written = os.path.abspath(out_path)
    except Exception as exc:  # noqa: BLE001
        written = None
        print(f"[WARN] could not write JSON: {exc}", file=sys.stderr)

    rec = facts["recommendation"]
    gpus = gpu.get("gpus") or []
    g = gpus[0] if gpus else {}
    dg = disk_facts.get("docker_root") or {}
    print("")
    print("=== inference-exp host survey ===")
    print(f"host        : {facts['host']['hostname']}  ({facts['os']['distro'].get('pretty') or facts['os']['system']})")
    print(f"kernel/arch : {facts['os']['release']} / {facts['os']['arch']}")
    print(f"cpu         : {facts['cpu']['logical']} logical / load {facts['cpu']['loadavg']}")
    print(
        "memory      : total {:.1f} GiB, available {:.1f} GiB, swap {:.1f} GiB".format(
            (mem_facts.get("total_mib") or 0) / 1024,
            (mem_facts.get("available_mib") or 0) / 1024,
            (mem_facts.get("swap_total_mib") or 0) / 1024,
        )
    )
    print(f"disk(docker): {dg.get('free_gb')} GiB free of {dg.get('total_gb')} GiB")
    if g:
        print(
            "gpu         : {} x{}, {:.1f} GiB total / {:.1f} GiB used, temp {}C, driver {}".format(
                g.get("name"),
                g.get("index"),
                (g.get("memory_total_mib") or 0) / 1024,
                (g.get("memory_used_mib") or 0) / 1024,
                g.get("temperature_c"),
                gpu.get("driver_version"),
            )
        )
        print(f"gpu detail  : cc={g.get('compute_cap')} pstate={g.get('pstate')} power={g.get('power_w')}W")
    else:
        print("gpu         : NOT VISIBLE")
    print(
        "docker      : {} {} nvidia-runtime={}".format(
            docker.get("server_version") or "?", docker.get("storage_driver") or "?", docker.get("nvidia_runtime")
        )
    )
    print(f"shm         : {facts['shm']}  nofile={facts.get('no_file_limit')}")
    if facts["network"]:
        for name, item in facts["network"].items():
            https = item.get("https", {})
            print(f"net {name:14s}: ok={https.get('ok')} status={https.get('status')} {https.get('ms')}ms")
    print("")
    print("--- recommendation ---")
    for key in (
        "gpu_memory_utilization",
        "container_memory_gb",
        "container_cpus",
        "max_model_len",
        "estimated_kv_budget_mib_qwen3_4b",
        "estimated_kv_tokens_fp16_qwen3_4b",
        "estimated_kv_tokens_fp8_qwen3_4b",
    ):
        print(f"{key:44s}: {rec.get(key)}")
    print("")
    if rec.get("issues"):
        for issue in rec["issues"]:
            print(_fmt_issue(issue))
    else:
        print("[ OK ] no blocking issues detected")
    if written:
        print("")
        print(f"JSON: {written}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
