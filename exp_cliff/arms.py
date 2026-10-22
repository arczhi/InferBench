"""Arm and level definitions for the concurrency-cliff experiment."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from common.config import Config
from common.runner import Arm, LevelPlan

PROFILES: Dict[str, Dict[str, Any]] = {
    "smoke": {
        "context": 8192,
        "turns": 3,
        "levels": [1, 2, 4],
        "max_tokens": 48,
        "engines": ["vllm", "sglang"],
    },
    "quick": {
        "context": 16000,
        "turns": 6,
        "levels": [1, 2, 4, 8, 12, 16],
        "max_tokens": 64,
        "engines": ["vllm", "sglang"],
    },
    "full": {
        "context": 22700,
        "turns": 12,
        "levels": [1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 32, 48],
        "max_tokens": 64,
        "engines": ["vllm", "sglang"],
    },
}


def build_arms(
    engines: Sequence[str],
    profile: str = "quick",
    cap: int = 12,
    only: Optional[Sequence[str]] = None,
) -> List[Arm]:
    arms: List[Arm] = []
    if "vllm" in engines:
        arms.extend(
            [
                Arm(
                    name="vllm-baseline",
                    engine="vllm",
                    engine_args=["--enable-prefix-caching", "--kv-cache-dtype", "auto"],
                    notes="default admission, bf16 KV, APC on",
                ),
                Arm(
                    name=f"vllm-cap{cap}",
                    engine="vllm",
                    engine_args=[
                        "--enable-prefix-caching",
                        "--kv-cache-dtype",
                        "auto",
                        "--max-num-seqs",
                        str(cap),
                    ],
                    notes=f"admission control: at most {cap} running sequences",
                ),
                Arm(
                    name="vllm-fp8",
                    engine="vllm",
                    engine_args=["--enable-prefix-caching", "--kv-cache-dtype", "fp8"],
                    notes="FP8 KV doubles cache capacity; no admission cap",
                ),
            ]
        )
        if profile == "full":
            arms.append(
                Arm(
                    name="vllm-noapc",
                    engine="vllm",
                    engine_args=["--no-enable-prefix-caching", "--kv-cache-dtype", "auto"],
                    notes="control: prefix reuse disabled (capped at N=8 to bound recompute cost)",
                    max_concurrency=8,
                )
            )
    if "sglang" in engines:
        arms.extend(
            [
                Arm(
                    name="sglang-baseline",
                    engine="sglang",
                    engine_args=[],
                    notes="RadixAttention, default scheduling",
                ),
                Arm(
                    name="sglang-fp8",
                    engine="sglang",
                    engine_args=["--kv-cache-dtype", "fp8_e5m2"],
                    notes="FP8 KV with RadixAttention",
                ),
            ]
        )
        if profile in ("quick", "full"):
            arms.append(
                Arm(
                    name=f"sglang-cap{cap}",
                    engine="sglang",
                    engine_args=["--max-running-requests", str(cap)],
                    notes=f"admission control: {cap} running requests",
                )
            )
    if profile == "smoke":
        keep = {"vllm-baseline", "sglang-baseline"}
        arms = [arm for arm in arms if arm.name in keep]
    if only:
        wanted = set(only)
        arms = [arm for arm in arms if arm.name in wanted]
    return arms


def build_levels(
    context: int,
    concurrencies: Sequence[int],
    turns: int,
    max_tokens: int,
    mode: str,
) -> List[LevelPlan]:
    if mode == "agent":
        system_tokens = int(context * 0.45)
        growth = int(context * 0.90) - system_tokens
        tool_result_tokens = max(64, int(growth / max(1, turns) - 220))
    else:
        system_tokens = int(context * 0.85)
        tool_result_tokens = 0
    return [
        LevelPlan(
            concurrency=concurrency,
            turns=turns,
            system_tokens=system_tokens,
            tool_result_tokens=tool_result_tokens,
            max_tokens=max_tokens,
            mode=mode,
            label=f"agent-{context}" if mode == "agent" else f"chat-{context}",
            context_hint=context,
        )
        for concurrency in concurrencies
    ]


def analyze_arm(levels: List[Dict[str, Any]], cfg: Config) -> Dict[str, Any]:
    rows = sorted(levels, key=lambda r: r["concurrency"])
    if not rows:
        return {}
    first = rows[0]
    last = rows[-1]
    baseline_p99 = (first.get("ttft_ms") or {}).get("p99")

    eviction_levels = []
    for row in rows:
        engine = row.get("engine") or {}
        evidence = {
            "concurrency": row["concurrency"],
            "cache_usage_peak": engine.get("cache_usage_peak"),
            "waiting_peak": engine.get("waiting_peak"),
            "running_peak": engine.get("running_peak"),
            "preemptions_delta": engine.get("preemptions_delta"),
            "apc_hit_rate_metric": engine.get("apc_hit_rate_metric"),
        }
        if (
            (engine.get("preemptions_delta") or 0) > 0
            or ((engine.get("cache_usage_peak") or 0) >= 0.95 and (engine.get("waiting_peak") or 0) > 0)
        ):
            eviction_levels.append(evidence)

    phase_breakdown = {}
    for row in rows:
        phases = row.get("phases") or {}
        phase_breakdown[str(row["concurrency"])] = {
            "tool_ttft_p99_ms": ((phases.get("tool") or {}).get("ttft_ms") or {}).get("p99"),
            "final_ttft_p99_ms": ((phases.get("final") or {}).get("ttft_ms") or {}).get("p99"),
            "agent_roundtrip_p99_ms": (phases.get("agent_roundtrip_e2e_ms") or {}).get("p99"),
            "cache_hit_rate_usage": row.get("cache_hit_rate_usage"),
        }

    return {
        "baseline_ttft_p99_ms": baseline_p99,
        "last_level_ttft_p99_ms": (last.get("ttft_ms") or {}).get("p99"),
        "ttft_growth_factor": round(
            ((last.get("ttft_ms") or {}).get("p99") or 0) / baseline_p99, 2
        )
        if baseline_p99
        else None,
        "eviction_evidence": eviction_levels,
        "phase_breakdown": phase_breakdown,
        "safe_levels": [r["concurrency"] for r in rows if r.get("safe")],
    }
