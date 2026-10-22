from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional

from .util import ROOT, clamp

DEFAULT_CONFIG_PATH = os.path.join(ROOT, "configs", "lab.json")


@dataclass
class DockerConfig:
    mirror: str = "docker.m.daocloud.io"
    vllm_image: str = "docker.m.daocloud.io/vllm/vllm-openai:latest"
    sglang_image: str = "docker.m.daocloud.io/lmsysorg/sglang:latest"
    hf_cache_volume: str = "inference-exp-hf"
    container_memory_gb: int = 24
    cpus: int = 6
    cpu_shares: int = 512
    shm_size_gb: int = 8
    pids_limit: int = 4096
    gpu_device: str = "0"
    python_bin: str = "python3"
    log_max_size: str = "200m"
    log_max_file: int = 2


@dataclass
class EngineConfig:
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 32768
    startup_timeout_s: int = 1800
    request_timeout_s: int = 900
    health_interval_s: int = 5
    metrics_interval_s: int = 2
    dtype: str = "bfloat16"
    vllm_port: int = 8000
    sglang_port: int = 30000


@dataclass
class GuardConfig:
    gpu_temp_warn_c: float = 80.0
    gpu_temp_abort_c: float = 87.0
    gpu_mem_used_abort_pct: float = 97.0
    host_mem_avail_abort_mib: int = 1536
    load_per_core_abort: float = 2.0
    disk_free_abort_gb: float = 5.0
    error_rate_abort: float = 0.5
    error_window: int = 20
    max_wall_s: int = 7200
    cooldown_max_s: int = 180
    cooldown_target_temp_c: float = 68.0
    sample_interval_s: float = 1.0


@dataclass
class WorkloadConfig:
    system_tokens: int = 10000
    tool_result_tokens: int = 900
    max_tokens: int = 64
    turns: int = 12
    warmup_tokens: int = 2048
    request_timeout_s: int = 900


@dataclass
class SLOConfig:
    ttft_p99_ms: float = 5000.0
    tpot_p99_ms: float = 150.0
    error_rate: float = 0.05


@dataclass
class Config:
    model: str = "Qwen/Qwen3-4B-Instruct-2507"
    served_model_name: str = "exp-model"
    hf_endpoint: str = ""
    docker: DockerConfig = field(default_factory=DockerConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    guard: GuardConfig = field(default_factory=GuardConfig)
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    slo: SLOConfig = field(default_factory=SLOConfig)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        def convert(obj: Any) -> Any:
            if hasattr(obj, "__dataclass_fields__"):
                return {f.name: convert(getattr(obj, f.name)) for f in fields(obj)}
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        return convert(self)


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _apply_env(raw: Dict[str, Any]) -> Dict[str, Any]:
    mapping = {
        "EXP_MODEL": ("model", str),
        "EXP_SERVED_MODEL_NAME": ("served_model_name", str),
        "EXP_HF_ENDPOINT": ("hf_endpoint", str),
        "EXP_GPU_MEM_UTIL": ("engine", "gpu_memory_utilization", float),
        "EXP_MAX_MODEL_LEN": ("engine", "max_model_len", int),
        "EXP_MAX_WALL_S": ("guard", "max_wall_s", int),
        "EXP_VLLM_PORT": ("engine", "vllm_port", int),
        "EXP_SGLANG_PORT": ("engine", "sglang_port", int),
        "EXP_CONTAINER_MEM_GB": ("docker", "container_memory_gb", int),
        "EXP_CPUS": ("docker", "cpus", int),
        "EXP_GPU_DEVICE": ("docker", "gpu_device", str),
        "VLLM_IMAGE": ("docker", "vllm_image", str),
        "SGLANG_IMAGE": ("docker", "sglang_image", str),
    }
    for env_key, spec in mapping.items():
        if env_key not in os.environ:
            continue
        value: Any = os.environ[env_key]
        try:
            if len(spec) == 2:
                path, caster = spec
                raw[path] = caster(value)
            else:
                section, path, caster = spec
                raw.setdefault(section, {})[path] = caster(value)
        except (TypeError, ValueError):
            raise SystemExit(f"invalid {env_key}={value!r}")
    return raw


def _host_ram_gb() -> Optional[float]:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1024.0 / 1024.0
    except Exception:  # noqa: BLE001
        pass
    return None


def load(path: Optional[str] = None) -> Config:
    path = path or os.environ.get("EXP_CONFIG") or DEFAULT_CONFIG_PATH
    raw: Dict[str, Any] = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    defaults = Config().to_dict()
    merged = _apply_env(_merge(defaults, raw))
    known = {f.name for f in fields(Config)}
    extra = {k: v for k, v in merged.items() if k not in known}
    kwargs = {k: v for k, v in merged.items() if k in known}
    cfg = Config(
        model=kwargs.get("model", defaults["model"]),
        served_model_name=kwargs.get("served_model_name", defaults["served_model_name"]),
        hf_endpoint=kwargs.get("hf_endpoint", ""),
        docker=DockerConfig(**kwargs.get("docker", {})),
        engine=EngineConfig(**kwargs.get("engine", {})),
        guard=GuardConfig(**kwargs.get("guard", {})),
        workload=WorkloadConfig(**kwargs.get("workload", {})),
        slo=SLOConfig(**kwargs.get("slo", {})),
        extra=extra,
    )
    adjust_for_host(cfg)
    return cfg


def adjust_for_host(cfg: Config) -> None:
    """Never let the default container limits exceed the host's real RAM."""
    ram_gb = _host_ram_gb()
    if ram_gb is None:
        return
    safe = int(clamp(ram_gb * 0.45, 6, 64))
    if cfg.docker.container_memory_gb > safe:
        print(
            f"[WARN] container memory {cfg.docker.container_memory_gb}GiB > host-safe {safe}GiB "
            f"(host {ram_gb:.1f}GiB); clamping"
        )
        cfg.docker.container_memory_gb = safe
    safe_cpus = max(2, (os.cpu_count() or 4) // 2)
    if cfg.docker.cpus > safe_cpus:
        print(
            f"[WARN] container cpus {cfg.docker.cpus} > half of host cores ({safe_cpus}); clamping"
        )
        cfg.docker.cpus = safe_cpus
    shm_cap = max(2, cfg.docker.container_memory_gb // 2)
    if cfg.docker.shm_size_gb > shm_cap:
        print(
            f"[WARN] shm {cfg.docker.shm_size_gb}GiB > half of container memory; "
            f"clamping to {shm_cap}GiB"
        )
        cfg.docker.shm_size_gb = shm_cap
