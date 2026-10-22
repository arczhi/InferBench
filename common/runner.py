from __future__ import annotations

import json
import os
import platform
import re
import socket
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .client import build_payload, chat_nonstream, chat_stream
from .config import Config
from .engine import (
    DockerEngine,
    EngineError,
    EngineStartError,
    ExternalEngine,
    cleanup_orphans,
    image_for,
)
from .gpu_metrics import GpuSampler
from .guard import Guard
from .host_metrics import HostSampler
from .stats import safe_div, summarize
from .util import (
    RUNS_DIR,
    JsonlWriter,
    atomic_write_json,
    ensure_dir,
    local_stamp,
    run_cmd,
    utc_now,
)
from .workload import (
    build_history_block,
    build_session,
    estimate_messages_tokens,
    estimate_tokens,
)


@dataclass
class Arm:
    name: str
    engine: str
    engine_args: List[str] = field(default_factory=list)
    workload: str = "agent"
    notes: str = ""
    max_concurrency: Optional[int] = None


@dataclass
class LevelPlan:
    concurrency: int
    turns: int
    system_tokens: int
    tool_result_tokens: int
    max_tokens: int
    mode: str = "agent"
    label: str = ""
    context_hint: int = 0
    warm_prefix: bool = False


class EngineSampler(threading.Thread):
    def __init__(self, engine: Any, out_path: str, interval_s: float, state: Dict[str, Any]) -> None:
        super().__init__(daemon=True, name="engine-sampler")
        self.engine = engine
        self.interval_s = interval_s
        self.state = state
        self.writer = JsonlWriter(out_path)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._records: List[Dict[str, Any]] = []

    def run(self) -> None:
        while not self._stop_event.is_set():
            started = time.monotonic()
            metrics = self.engine.metrics()
            record = {
                "ts": utc_now(),
                "mono": time.monotonic(),
                "level": self.state.get("level"),
                "phase": self.state.get("phase"),
                "ok": bool(metrics),
                "metrics": metrics,
            }
            with self._lock:
                self._records.append(record)
                if len(self._records) > 8000:
                    del self._records[:1000]
            self.writer.write(record)
            elapsed = time.monotonic() - started
            self._stop_event.wait(max(0.2, self.interval_s - elapsed))

    def window(self, level: int) -> List[Dict[str, Any]]:
        with self._lock:
            return [r for r in self._records if r.get("level") == level and r.get("ok")]

    def stop(self) -> None:
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=10)
        self.writer.close()


def metric_value(metrics: Dict[str, float], names: Sequence[str]) -> Optional[float]:
    for name in names:
        if name in metrics:
            return metrics[name]
    return None


def _metric_series(samples: Sequence[Dict[str, Any]], names: Sequence[str]) -> List[float]:
    out: List[float] = []
    for sample in samples:
        value = metric_value(sample.get("metrics") or {}, names)
        if value is not None:
            out.append(float(value))
    return out


CACHE_USAGE = ["vllm:gpu_cache_usage_perc", "sglang:token_usage"]
WAITING = ["vllm:num_requests_waiting", "sglang:num_queue_reqs"]
RUNNING = ["vllm:num_requests_running", "sglang:num_running_reqs"]
PREEMPTIONS = [
    "vllm:num_preemptions_total",
    "vllm:num_preemptions",
    "sglang:num_retracted_reqs_total",
    "sglang:num_retracted_reqs",
]
APC_HITS = ["vllm:prefix_cache_hits_total", "sglang:cached_tokens_total"]
APC_QUERIES = ["vllm:prefix_cache_queries_total", "sglang:prompt_tokens_total"]
PROMPT_TOKENS = ["vllm:prompt_tokens_total", "sglang:prompt_tokens_total"]
GENERATION_TOKENS = ["vllm:generation_tokens_total", "sglang:generation_tokens_total"]
CACHE_HIT_RATE = ["sglang:cache_hit_rate"]


def level_engine_stats(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        return {}
    cache = _metric_series(samples, CACHE_USAGE)
    waiting = _metric_series(samples, WAITING)
    running = _metric_series(samples, RUNNING)
    hit_rate = _metric_series(samples, CACHE_HIT_RATE)
    pre = _metric_series(samples, PREEMPTIONS)
    hits = _metric_series(samples, APC_HITS)
    queries = _metric_series(samples, APC_QUERIES)
    prompt = _metric_series(samples, PROMPT_TOKENS)
    generation = _metric_series(samples, GENERATION_TOKENS)
    out: Dict[str, Any] = {
        "samples": len(samples),
        "cache_usage_peak": round(max(cache), 4) if cache else None,
        "cache_usage_mean": round(sum(cache) / len(cache), 4) if cache else None,
        "waiting_peak": max(waiting) if waiting else None,
        "waiting_mean": round(sum(waiting) / len(waiting), 2) if waiting else None,
        "running_peak": max(running) if running else None,
        "running_mean": round(sum(running) / len(running), 2) if running else None,
        "cache_hit_rate_metric": round(sum(hit_rate) / len(hit_rate), 4) if hit_rate else None,
    }
    if len(pre) >= 2:
        out["preemptions_delta"] = round(pre[-1] - pre[0], 1)
    if len(hits) >= 2 and len(queries) >= 2:
        dq = queries[-1] - queries[0]
        if dq > 0:
            out["apc_hit_rate_metric"] = round((hits[-1] - hits[0]) / dq, 4)
    if len(prompt) >= 2:
        out["engine_prompt_tokens_delta"] = round(prompt[-1] - prompt[0], 0)
    if len(generation) >= 2:
        out["engine_generation_tokens_delta"] = round(generation[-1] - generation[0], 0)
    return out


def cliff_analysis(level_rows: List[Dict[str, Any]], slo: Any) -> Dict[str, Any]:
    if not level_rows:
        return {}
    ordered = sorted(level_rows, key=lambda r: r["concurrency"])
    baseline = ordered[0]
    base_ttft = (baseline.get("ttft_ms") or {}).get("p99")
    threshold = max(slo.ttft_p99_ms, (base_ttft or 0) * 5.0)
    first_unsafe: Optional[Dict[str, Any]] = None
    for row in ordered:
        ttft_p99 = (row.get("ttft_ms") or {}).get("p99")
        tpot_p99 = (row.get("tpot_ms") or {}).get("p99")
        error_rate = row.get("error_rate") or 0.0
        unsafe = (
            (ttft_p99 is not None and ttft_p99 > threshold)
            or (tpot_p99 is not None and tpot_p99 > slo.tpot_p99_ms)
            or error_rate > slo.error_rate
            or ((row.get("engine") or {}).get("waiting_peak") or 0) > 0
            and (row.get("ttft_ms") or {}).get("p99", 0) > slo.ttft_p99_ms
        )
        if unsafe:
            first_unsafe = row
            break
    peak = max(ordered, key=lambda r: r.get("output_tok_s") or 0)
    collapse = None
    for row in ordered:
        if row["concurrency"] > peak["concurrency"] and (row.get("output_tok_s") or 0) < 0.8 * (peak.get("output_tok_s") or 0):
            collapse = row
            break
    return {
        "baseline_concurrency": baseline["concurrency"],
        "baseline_ttft_p99_ms": base_ttft,
        "unsafe_threshold_ms": round(threshold, 1),
        "cliff_concurrency": first_unsafe["concurrency"] if first_unsafe else None,
        "cliff_evidence": {
            "ttft_p99_ms": (first_unsafe.get("ttft_ms") or {}).get("p99"),
            "tpot_p99_ms": (first_unsafe.get("tpot_ms") or {}).get("p99"),
            "error_rate": first_unsafe.get("error_rate"),
            "waiting_peak": (first_unsafe.get("engine") or {}).get("waiting_peak"),
            "cache_usage_peak": (first_unsafe.get("engine") or {}).get("cache_usage_peak"),
            "preemptions_delta": (first_unsafe.get("engine") or {}).get("preemptions_delta"),
        }
        if first_unsafe
        else None,
        "max_safe_concurrency": max(
            [r["concurrency"] for r in ordered if r.get("safe")] or [0]
        ),
        "throughput_peak": {
            "concurrency": peak["concurrency"],
            "output_tok_s": peak.get("output_tok_s"),
        },
        "throughput_collapse": {
            "concurrency": collapse["concurrency"],
            "output_tok_s": collapse.get("output_tok_s"),
        }
        if collapse
        else None,
    }


def _phase_stats(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    phases: Dict[str, Any] = {}
    for phase in ("tool", "final", "chat"):
        selected = [r for r in records if r.get("phase") == phase and r.get("ok")]
        phases[phase] = {
            "requests": len(selected),
            "ttft_ms": summarize([r["ttft_ms"] for r in selected if r.get("ttft_ms") is not None]),
            "e2e_ms": summarize([r["e2e_ms"] for r in selected if r.get("e2e_ms") is not None]),
            "completion_tokens": sum(r.get("completion_tokens") or 0 for r in selected),
        }
    pairs: Dict[Any, Dict[str, Dict[str, Any]]] = {}
    for record in records:
        if not record.get("ok") or record.get("phase") not in ("tool", "final"):
            continue
        key = (record.get("session_id"), record.get("turn"))
        pairs.setdefault(key, {})[record["phase"]] = record
    roundtrips = [
        pair["tool"]["e2e_ms"] + pair["final"]["e2e_ms"]
        for pair in pairs.values()
        if "tool" in pair and "final" in pair
    ]
    phases["agent_roundtrip_e2e_ms"] = summarize(roundtrips)
    return phases


class ExperimentRunner:
    def __init__(
        self,
        cfg: Config,
        experiment: str,
        arms: List[Arm],
        levels: List[LevelPlan],
        out_root: Optional[str] = None,
        external_url: Optional[str] = None,
        level_timeout_s: int = 1800,
        analyzer: Optional[Callable[[List[Dict[str, Any]], Config], Dict[str, Any]]] = None,
    ) -> None:
        self.cfg = cfg
        self.experiment = experiment
        self.arms = arms
        self.levels = list(levels)
        self.out_root = out_root or os.path.join(RUNS_DIR, experiment)
        self.external_url = external_url
        self.level_timeout_s = level_timeout_s
        self.analyzer = analyzer
        self.run_id = f"{experiment}-{local_stamp()}-{os.getpid()}"
        self.run_dir = ensure_dir(os.path.join(self.out_root, self.run_id))
        self._lock_fh: Optional[Any] = None
        self._aborted = False
        self._arm_summaries: List[Dict[str, Any]] = []
        self.model_name = cfg.served_model_name
        self.docker_root: Optional[str] = None

    # ------------------------------------------------------------------- setup
    def _lock(self) -> None:
        try:
            import fcntl

            ensure_dir(RUNS_DIR)
            path = os.path.join(RUNS_DIR, ".lock")
            fh = open(path, "a+")
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise SystemExit(f"another experiment is running (lock: {path})")
            fh.write(f"{os.getpid()} {self.run_id}\n")
            fh.flush()
            self._lock_fh = fh
        except ImportError:
            pass

    def _unlock(self) -> None:
        if self._lock_fh is not None:
            try:
                import fcntl

                fcntl.flock(self._lock_fh.fileno(), fcntl.LOCK_UN)
                self._lock_fh.close()
            except Exception:  # noqa: BLE001
                pass
            self._lock_fh = None

    def _preflight(self) -> None:
        print(f"[{utc_now()}] run: {self.run_id}")
        print(f"[{utc_now()}] model: {self.cfg.model}  served={self.cfg.served_model_name}")
        print(
            f"[{utc_now()}] gpu_mem_util={self.cfg.engine.gpu_memory_utilization} "
            f"max_model_len={self.cfg.engine.max_model_len} "
            f"container_mem={self.cfg.docker.container_memory_gb}GiB cpus={self.cfg.docker.cpus}"
        )
        if self.external_url:
            print(f"[{utc_now()}] external engine mode: {self.external_url} (guard cannot stop it)")
            atomic_write_json(
                os.path.join(self.run_dir, "run_meta.json"),
                {
                    "run_id": self.run_id,
                    "experiment": self.experiment,
                    "created_at": utc_now(),
                    "config": self.cfg.to_dict(),
                    "external_url": self.external_url,
                    "arms": [arm.__dict__ for arm in self.arms],
                    "levels": [level.__dict__ for level in self.levels],
                    "host": {"hostname": socket.gethostname(), "python": platform.python_version()},
                },
            )
            return
        if not self.cfg.model:
            raise SystemExit("model is empty")
        if not (0.1 <= self.cfg.engine.gpu_memory_utilization <= 0.95):
            raise SystemExit("gpu_memory_utilization must be between 0.1 and 0.95")
        if self.cfg.engine.max_model_len < 2048:
            raise SystemExit("max_model_len is too small for this experiment")
        docker_root = run_cmd(["docker", "info", "--format", "{{.DockerRootDir}}"], timeout=10)
        if docker_root["ok"] and docker_root["out"]:
            self.docker_root = docker_root["out"]
        try:
            import shutil

            probe_path = self.docker_root or self.run_dir
            free_gb = shutil.disk_usage(probe_path).free / 2**30
            if free_gb < self.cfg.guard.disk_free_abort_gb:
                raise SystemExit(f"only {free_gb:.1f} GiB free under {probe_path}")
        except OSError:
            pass
        gpu = run_cmd(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], timeout=10)
        if not gpu["ok"]:
            raise SystemExit(f"nvidia-smi failed: {gpu['err']}")
        print(f"[{utc_now()}] gpu: {gpu['out'].splitlines()[0] if gpu['out'] else '?'}")
        docker = run_cmd(["docker", "info", "--format", "{{.ServerVersion}}"], timeout=15)
        if not docker["ok"]:
            raise SystemExit(f"docker daemon not reachable: {docker['err']}")
        for kind in sorted({arm.engine for arm in self.arms}):
            image = image_for(self.cfg, kind)
            if not run_cmd(["docker", "image", "inspect", image], timeout=20)["ok"]:
                print(f"[WARN] image missing for {kind}: {image} (arms will be skipped; run scripts/pull_images.sh)")
        cleanup_orphans(self.run_id)
        atomic_write_json(
            os.path.join(self.run_dir, "run_meta.json"),
            {
                "run_id": self.run_id,
                "experiment": self.experiment,
                "created_at": utc_now(),
                "config": self.cfg.to_dict(),
                "external_url": self.external_url,
                "arms": [arm.__dict__ for arm in self.arms],
                "levels": [level.__dict__ for level in self.levels],
                "host": {
                    "hostname": socket.gethostname(),
                    "python": platform.python_version(),
                },
            },
        )

    # --------------------------------------------------------------- resources
    def _make_engine(self, arm: Arm, arm_dir: str) -> Any:
        if self.external_url:
            return ExternalEngine(self.external_url, arm.engine, self.cfg, arm_dir)
        return DockerEngine(
            self.cfg,
            arm.engine,
            arm.name,
            self.run_id,
            arm_dir,
            arm.engine_args,
        )

    def _adopt_served_name(self, info: Dict[str, Any]) -> None:
        data = ((info.get("models") or {}).get("data") or []) if isinstance(info.get("models"), dict) else []
        if data and isinstance(data[0], dict) and data[0].get("id"):
            self.model_name = str(data[0]["id"])
            if self.model_name != self.cfg.served_model_name:
                print(f"[{utc_now()}] using served model id from engine: {self.model_name}")

    def _warmup(self, engine: Any, arm: Arm) -> Dict[str, Any]:
        cfg = self.cfg
        records: List[Dict[str, Any]] = []
        salt = f"warmup-{self.run_id}"
        short = build_session(
            level=-1,
            session_id=-1,
            salt=salt,
            mode="chat",
            system_tokens=min(1024, cfg.workload.system_tokens),
            turns=1,
            tool_result_tokens=64,
            seed=11,
        )
        messages = short.build_messages(0, "chat", [], cfg.engine.max_model_len - 64)
        started = time.monotonic()
        res = chat_nonstream(
            engine.base_url,
            build_payload(self.model_name, messages, max_tokens=8, stream=False),
            timeout_s=cfg.engine.request_timeout_s,
        )
        records.append({"phase": "short", "ok": res.ok, "ms": round(1000 * (time.monotonic() - started), 1), "error": res.error})

        long_tokens = max(2048, cfg.workload.system_tokens)
        if long_tokens >= 2048:
            long_session = build_session(
                level=-1,
                session_id=-1,
                salt=salt + "-long",
                mode="chat",
                system_tokens=long_tokens,
                turns=1,
                tool_result_tokens=cfg.workload.tool_result_tokens,
                seed=12,
            )
            messages = long_session.build_messages(
                0, "chat", [], cfg.engine.max_model_len - cfg.workload.max_tokens - 256
            )
            planned = estimate_messages_tokens(messages)
            started = time.monotonic()
            res = chat_nonstream(
                engine.base_url,
                build_payload(self.model_name, messages, max_tokens=8, stream=False),
                timeout_s=cfg.engine.request_timeout_s,
            )
            records.append(
                {
                    "phase": "long",
                    "planned_tokens": planned,
                    "actual_prompt_tokens": res.prompt_tokens,
                    "ok": res.ok,
                    "ms": round(1000 * (time.monotonic() - started), 1),
                    "error": res.error,
                }
            )
        atomic_write_json(os.path.join(engine.run_dir, "warmup.json"), records)
        if not all(r.get("ok") for r in records):
            raise EngineError(f"warmup failed: {records}")
        return {"records": records}

    # ---------------------------------------------------------------- workload
    def _run_level(
        self, engine: Any, arm: Arm, plan: LevelPlan, level_index: int, guard: Guard
    ) -> Dict[str, Any]:
        cfg = self.cfg
        salt = f"{arm.name}-L{plan.concurrency}"
        specs = [
            build_session(
                level=plan.concurrency,
                session_id=i,
                salt=salt,
                mode=plan.mode,
                system_tokens=plan.system_tokens,
                turns=plan.turns,
                tool_result_tokens=plan.tool_result_tokens,
                seed=17,
            )
            for i in range(plan.concurrency)
        ]
        arm_dir = engine.run_dir
        req_writer = JsonlWriter(os.path.join(arm_dir, "requests.jsonl"))
        sess_writer = JsonlWriter(os.path.join(arm_dir, "sessions.jsonl"))
        records: List[Dict[str, Any]] = []
        sessions: List[Dict[str, Any]] = []
        rec_lock = threading.Lock()
        start_gun = threading.Event()
        stop_level = threading.Event()
        prefix_warmup: Optional[Dict[str, Any]] = None
        if plan.warm_prefix:
            warm_spec = build_session(
                level=plan.concurrency,
                session_id=-1,
                salt=salt,
                mode=plan.mode,
                system_tokens=plan.system_tokens,
                turns=1,
                tool_result_tokens=plan.tool_result_tokens,
                seed=19,
            )
            warm_phase = "tool" if plan.mode == "agent" else "chat"
            warm_messages = warm_spec.build_messages(
                0, warm_phase, [], cfg.engine.max_model_len - 8 - 256
            )
            warm_started = time.monotonic()
            warm_res = chat_nonstream(
                engine.base_url,
                build_payload(self.model_name, warm_messages, max_tokens=8, stream=False),
                timeout_s=cfg.engine.request_timeout_s,
            )
            prefix_warmup = {
                "ok": warm_res.ok,
                "error": warm_res.error,
                "ms": round(1000 * (time.monotonic() - warm_started), 1),
                "prompt_tokens": warm_res.prompt_tokens,
                "cached_tokens": warm_res.cached_tokens,
                "planned_tokens": estimate_messages_tokens(warm_messages),
            }
            if not warm_res.ok:
                print(f"[{utc_now()}] prefix warmup failed at N={plan.concurrency}: {warm_res.error}")
        started_mono = time.monotonic()
        deadline = started_mono + self.level_timeout_s

        def one_request(spec: Any, turn_idx: int, phase: str, history: Any, tool_call_text: Optional[str], max_tokens: int) -> Any:
            budget = cfg.engine.max_model_len - max_tokens - 256
            messages = spec.build_messages(turn_idx, phase, history, budget, tool_call_text)
            planned = estimate_messages_tokens(messages)
            payload = build_payload(self.model_name, messages, max_tokens=max_tokens, stream=True)
            t0 = time.monotonic()
            res = chat_stream(engine.base_url, payload, timeout_s=cfg.engine.request_timeout_s)
            t1 = time.monotonic()
            tpot = None
            if res.ok and res.ttft_ms is not None and res.completion_tokens and res.completion_tokens > 1:
                tpot = round((res.e2e_ms - res.ttft_ms) / (res.completion_tokens - 1), 3)
            itl = summarize(res.itl_ms)
            record = {
                "run_id": self.run_id,
                "arm": arm.name,
                "engine": arm.engine,
                "level": plan.concurrency,
                "session_id": spec.session_id,
                "turn": turn_idx,
                "phase": phase,
                "ok": res.ok,
                "status": res.status,
                "error": res.error,
                "ttft_ms": res.ttft_ms,
                "ttft_event_ms": res.ttft_event_ms,
                "e2e_ms": res.e2e_ms,
                "tpot_ms": tpot,
                "itl_ms": itl,
                "prompt_tokens": res.prompt_tokens,
                "completion_tokens": res.completion_tokens,
                "cached_tokens": res.cached_tokens,
                "planned_tokens": planned,
                "finish_reason": res.finish_reason,
                "usage_estimated": res.usage_estimated,
                "started_mono": t0,
                "ended_mono": t1,
                "started_at": utc_now(),
            }
            with rec_lock:
                records.append(record)
            req_writer.write(record)
            return res

        def worker(spec: Any) -> None:
            start_gun.wait()
            history: List[Any] = []
            session_started = time.monotonic()
            errors: List[str] = []
            turns_completed = 0
            ttft_values: List[float] = []
            prompt_sum = 0
            completion_sum = 0
            for turn_idx, turn in enumerate(spec.turns):
                if stop_level.is_set():
                    break
                if spec.mode == "agent":
                    r1 = one_request(spec, turn_idx, "tool", history, None, plan.max_tokens)
                    if r1.ttft_ms is not None:
                        ttft_values.append(r1.ttft_ms)
                    if not r1.ok:
                        errors.append(f"turn {turn_idx} tool: {r1.error}")
                        break
                    tool_text = r1.text if r1.text else None
                    r2 = one_request(spec, turn_idx, "final", history, tool_text, plan.max_tokens)
                    if r2.ttft_ms is not None:
                        ttft_values.append(r2.ttft_ms)
                    if not r2.ok:
                        errors.append(f"turn {turn_idx} final: {r2.error}")
                        break
                    history.append(build_history_block(turn, turn_idx, r2.text, spec.mode, tool_text))
                    turns_completed += 1
                else:
                    r = one_request(spec, turn_idx, "chat", history, None, plan.max_tokens)
                    if r.ttft_ms is not None:
                        ttft_values.append(r.ttft_ms)
                    if not r.ok:
                        errors.append(f"turn {turn_idx}: {r.error}")
                        break
                    history.append(build_history_block(turn, turn_idx, r.text, spec.mode))
                    turns_completed += 1
            summary = {
                "run_id": self.run_id,
                "arm": arm.name,
                "engine": arm.engine,
                "level": plan.concurrency,
                "session_id": spec.session_id,
                "ok": turns_completed == plan.turns and not errors,
                "turns_completed": turns_completed,
                "planned_turns": plan.turns,
                "wall_s": round(time.monotonic() - session_started, 2),
                "errors": errors,
                "ttft_mean_ms": round(sum(ttft_values) / len(ttft_values), 2) if ttft_values else None,
            }
            with rec_lock:
                sessions.append(summary)
            sess_writer.write(summary)

        threads = [threading.Thread(target=worker, args=(spec,), daemon=True, name=f"session-{spec.session_id}") for spec in specs]
        for thread in threads:
            thread.start()
        time.sleep(0.25)
        start_gun.set()
        timed_out = False
        while any(thread.is_alive() for thread in threads):
            if guard.abort_event.is_set() or self._aborted or stop_level.is_set():
                break
            if time.monotonic() > deadline:
                timed_out = True
                print(f"[{utc_now()}] level {plan.concurrency} exceeded {self.level_timeout_s}s; stopping engine")
                stop_level.set()
                guard.stop()
                try:
                    engine.stop()
                except Exception:  # noqa: BLE001
                    pass
                break
            time.sleep(1.0)
        for thread in threads:
            thread.join(timeout=60)
        req_writer.close()
        sess_writer.close()

        ok_records = [r for r in records if r.get("ok")]
        ttft = [r["ttft_ms"] for r in ok_records if r.get("ttft_ms") is not None]
        tpot = [r["tpot_ms"] for r in ok_records if r.get("tpot_ms") is not None]
        e2e = [r["e2e_ms"] for r in ok_records if r.get("e2e_ms") is not None]
        prompt_tokens = sum(r.get("prompt_tokens") or 0 for r in ok_records)
        cached_tokens = sum(r.get("cached_tokens") or 0 for r in ok_records if r.get("cached_tokens") is not None)
        completion_tokens = sum(r.get("completion_tokens") or 0 for r in ok_records)
        starts = [r["started_mono"] for r in records if r.get("started_mono")]
        ends = [r["ended_mono"] for r in records if r.get("ended_mono")]
        wall = (max(ends) - min(starts)) if starts and ends else 0.0
        errors = len(records) - len(ok_records)
        error_rate = safe_div(errors, len(records)) or 0.0
        ttft_summary = summarize(ttft)
        tpot_summary = summarize(tpot)
        row = {
            "arm": arm.name,
            "engine": arm.engine,
            "workload": plan.mode,
            "label": plan.label,
            "context_hint": plan.context_hint,
            "concurrency": plan.concurrency,
            "sessions": len(sessions),
            "sessions_ok": sum(1 for s in sessions if s.get("ok")),
            "turns": plan.turns,
            "requests": len(records),
            "requests_ok": len(ok_records),
            "errors": errors,
            "error_rate": round(error_rate, 4),
            "wall_s": round(wall, 2),
            "ttft_ms": ttft_summary,
            "tpot_ms": tpot_summary,
            "e2e_ms": summarize(e2e),
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "completion_tokens": completion_tokens,
            "cache_hit_rate_usage": round(cached_tokens / prompt_tokens, 4) if prompt_tokens else None,
            "output_tok_s": round(completion_tokens / wall, 2) if wall else None,
            "prefill_tok_s": round(prompt_tokens / wall, 2) if wall else None,
            "total_tok_s": round((prompt_tokens + completion_tokens) / wall, 2) if wall else None,
            "engine": {},
            "phases": _phase_stats(records),
            "prefix_warmup": prefix_warmup,
            "timed_out": timed_out,
            "level_index": level_index,
        }
        row["safe"] = bool(
            row["requests_ok"] > 0
            and row["error_rate"] <= cfg.slo.error_rate
            and (ttft_summary.get("p99") is None or ttft_summary["p99"] <= cfg.slo.ttft_p99_ms)
            and (tpot_summary.get("p99") is None or tpot_summary["p99"] <= cfg.slo.tpot_p99_ms)
        )
        return row

    # ------------------------------------------------------------------- main
    def run(self) -> Dict[str, Any]:
        self._lock()
        started = time.monotonic()
        self._preflight()
        summary: Dict[str, Any] = {
            "run_id": self.run_id,
            "experiment": self.experiment,
            "started_at": utc_now(),
            "finished_at": None,
            "aborted": False,
            "abort_reason": None,
            "arms": [],
            "config": self.cfg.to_dict(),
        }
        try:
            for arm in self.arms:
                if self._aborted:
                    break
                arm_dir = ensure_dir(os.path.join(self.run_dir, re.sub(r"[^A-Za-z0-9_.-]+", "_", arm.name)))
                print(f"\n[{utc_now()}] === arm {arm.name} ({arm.engine}) ===")
                arm_result = self._run_arm(arm, arm_dir)
                summary["arms"].append(arm_result)
                self._arm_summaries.append(arm_result)
                if self._aborted:
                    break
        finally:
            summary["finished_at"] = utc_now()
            summary["duration_s"] = round(time.monotonic() - started, 1)
            summary["aborted"] = self._aborted
            summary["abort_reason"] = self._abort_reason if self._aborted else None
            atomic_write_json(os.path.join(self.run_dir, "summary.json"), summary)
            self._unlock()
        return summary

    def _run_arm(self, arm: Arm, arm_dir: str) -> Dict[str, Any]:
        cfg = self.cfg
        engine = self._make_engine(arm, arm_dir)
        samplers: List[Any] = []
        guard: Optional[Guard] = None
        watcher: Optional[threading.Thread] = None
        watch_stop = threading.Event()
        result: Dict[str, Any] = {
            "name": arm.name,
            "engine": arm.engine,
            "notes": arm.notes,
            "started_at": utc_now(),
            "status": "failed",
            "levels": [],
            "skipped_reason": None,
        }
        self._engine_samples: Dict[int, List[Dict[str, Any]]] = {}
        try:
            missing = engine.check_flags(engine.command())
            if "--enable-prefix-caching" in missing:
                print(f"[{utc_now()}] note: --enable-prefix-caching not listed in this image; relying on v1 default")
                arm.engine_args = [a for a in arm.engine_args if not a.startswith("--enable-prefix-caching")]
                missing = engine.check_flags(engine.command())
            missing = engine.apply_flag_fallbacks(missing)
            if missing:
                result["status"] = "skipped"
                result["skipped_reason"] = f"engine does not support flags: {missing}"
                print(f"[{utc_now()}] SKIP {arm.name}: {result['skipped_reason']}")
                atomic_write_json(os.path.join(arm_dir, "skipped.json"), result)
                return result

            gpu = GpuSampler(os.path.join(arm_dir, "gpu.jsonl"), interval_s=cfg.guard.sample_interval_s)
            host = HostSampler(
                os.path.join(arm_dir, "host.jsonl"),
                disk_path=self.docker_root or arm_dir,
                interval_s=cfg.guard.sample_interval_s,
            )
            gpu.start()
            host.start()
            samplers.extend([gpu, host])
            metric_state: Dict[str, Any] = {"level": None, "phase": "startup"}
            engine_sampler = EngineSampler(
                engine,
                os.path.join(arm_dir, "engine_metrics.jsonl"),
                interval_s=cfg.engine.metrics_interval_s,
                state=metric_state,
            )

            guard = Guard(cfg, arm_dir, gpu, host, engine.container_state, time.monotonic())
            guard.start()
            watcher = threading.Thread(target=self._abort_watcher, args=(guard, engine, watch_stop), daemon=True)
            watcher.start()

            engine.start()
            engine_sampler.start()
            samplers.append(engine_sampler)
            info = engine.server_info()
            self._adopt_served_name(info)
            atomic_write_json(os.path.join(arm_dir, "server_info.json"), info)
            atomic_write_json(os.path.join(arm_dir, "image.json"), engine.image_digest())
            result["server_info"] = info
            print(f"[{utc_now()}] server: {json.dumps(info, default=str)[:500]}")
            self._warmup(engine, arm)

            expected_levels = [
                plan for plan in self.levels
                if arm.max_concurrency is None or plan.concurrency <= arm.max_concurrency
            ]
            for index, plan in enumerate(self.levels):
                if arm.max_concurrency is not None and plan.concurrency > arm.max_concurrency:
                    continue
                if guard.abort_event.is_set():
                    break
                metric_state["level"] = index
                metric_state["phase"] = "running"
                print(
                    f"[{utc_now()}] {arm.name}: N={plan.concurrency} turns={plan.turns} "
                    f"mode={plan.mode} label={plan.label or '-'} system_tokens={plan.system_tokens}"
                )
                row = self._run_level(engine, arm, plan, index, guard)
                samples = engine_sampler.window(index)
                final_metrics = engine.metrics()
                if final_metrics:
                    samples = samples + [
                        {"ts": utc_now(), "mono": time.monotonic(), "level": index, "phase": "post", "ok": True, "metrics": final_metrics}
                    ]
                self._engine_samples[index] = samples
                row["engine"] = level_engine_stats(self._engine_samples[index])
                result["levels"].append(row)
                with open(os.path.join(arm_dir, "levels.jsonl"), "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, default=str) + "\n")
                metric_state["phase"] = "idle"
                if row.get("timed_out"):
                    print(f"[{utc_now()}] level N={plan.concurrency} timed out; aborting arm")
                    break
                if row["error_rate"] > cfg.slo.error_rate or row["requests_ok"] == 0:
                    print(f"[{utc_now()}] level N={plan.concurrency} is unhealthy; continuing to next arm")
                    break
                if guard.abort_event.is_set():
                    break

            if not result["levels"]:
                result["status"] = "failed"
            elif len(result["levels"]) == len(expected_levels):
                result["status"] = "completed"
            else:
                result["status"] = "partial"
            if guard.abort_event.is_set():
                self._aborted = True
                self._abort_reason = guard.reason
                result["abort_reason"] = guard.reason
            result["cliff"] = cliff_analysis(result["levels"], cfg.slo)
            if self.analyzer:
                try:
                    result["analysis"] = self.analyzer(result["levels"], cfg)
                except Exception as exc:  # noqa: BLE001
                    result["analysis_error"] = repr(exc)
        except (EngineStartError, EngineError) as exc:
            result["error"] = str(exc)
            result["traceback"] = traceback.format_exc()
            print(f"[{utc_now()}] arm {arm.name} failed: {exc}")
        except Exception as exc:  # noqa: BLE001
            result["error"] = repr(exc)
            result["traceback"] = traceback.format_exc()
            print(f"[{utc_now()}] arm {arm.name} crashed: {exc}\n{traceback.format_exc()}")
        finally:
            watch_stop.set()
            if watcher is not None:
                watcher.join(timeout=5)
            if guard is not None:
                guard.stop()
                if guard.abort_event.is_set():
                    self._aborted = True
                    self._abort_reason = guard.reason
            try:
                engine.stop()
            except Exception:  # noqa: BLE001
                pass
            for sampler in reversed(samplers):
                try:
                    sampler.stop()
                except Exception:  # noqa: BLE001
                    pass
            result["finished_at"] = utc_now()
            atomic_write_json(os.path.join(arm_dir, "arm_summary.json"), result)
            if not self._aborted and result.get("status") != "skipped":
                self._cooldown()
        return result

    def _abort_watcher(self, guard: Guard, engine: Any, watch_stop: threading.Event) -> None:
        while not watch_stop.is_set():
            if guard.abort_event.wait(0.5):
                print(f"[{utc_now()}] guard tripped; stopping engine")
                try:
                    engine.stop()
                except Exception:  # noqa: BLE001
                    pass
                return

    def _cooldown(self) -> None:
        cfg = self.cfg
        target = cfg.guard.cooldown_target_temp_c
        deadline = time.monotonic() + cfg.guard.cooldown_max_s
        while time.monotonic() < deadline:
            gpu = query_latest_gpu()
            if not gpu.get("ok"):
                return
            temp = gpu.get("temp_c")
            free = gpu.get("mem_free_mib")
            total = gpu.get("mem_total_mib")
            if temp is not None and temp <= target and free is not None and total and free > 0.85 * total:
                return
            time.sleep(5)
        print(f"[{utc_now()}] cooldown timeout; continuing")


def query_latest_gpu() -> Dict[str, Any]:
    from .gpu_metrics import query_gpu

    return query_gpu()
