"""Arm and level definitions for the KV-cache strategy experiment."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from common.config import Config
from common.runner import Arm, LevelPlan

PROFILES: Dict[str, Dict[str, Any]] = {
    "smoke": {
        "contexts": [8192],
        "levels": [1, 2],
        "max_tokens": 128,
        "decode_probe": False,
        "engines": ["vllm"],
    },
    "quick": {
        "contexts": [8192, 16384],
        "levels": [1, 2, 4, 8],
        "max_tokens": 192,
        "decode_probe": True,
        "engines": ["vllm", "sglang"],
    },
    "full": {
        "contexts": [8192, 16384, 22700],
        "levels": [1, 2, 4, 6, 8, 12, 16],
        "max_tokens": 256,
        "decode_probe": True,
        "engines": ["vllm", "sglang"],
    },
}


def build_arms(
    engines: Sequence[str],
    profile: str = "quick",
    include_hicache: bool = False,
    only: Optional[Sequence[str]] = None,
) -> List[Arm]:
    arms: List[Arm] = []
    if "vllm" in engines:
        arms.extend(
            [
                Arm(
                    name="vllm-fp16-apc",
                    engine="vllm",
                    engine_args=["--enable-prefix-caching", "--kv-cache-dtype", "auto"],
                    notes="baseline: bf16 KV + automatic prefix caching",
                ),
                Arm(
                    name="vllm-fp8-apc",
                    engine="vllm",
                    engine_args=["--enable-prefix-caching", "--kv-cache-dtype", "fp8"],
                    notes="FP8 KV cache, same APC; tests capacity gain and decode cost",
                ),
                Arm(
                    name="vllm-fp16-noapc",
                    engine="vllm",
                    engine_args=["--no-enable-prefix-caching", "--kv-cache-dtype", "auto"],
                    notes="control: no prefix reuse, isolates APC contribution",
                ),
            ]
        )
    if "sglang" in engines:
        arms.extend(
            [
                Arm(
                    name="sglang-fp16-radix",
                    engine="sglang",
                    engine_args=[],
                    notes="baseline: RadixAttention, bf16 KV",
                ),
                Arm(
                    name="sglang-fp8-radix",
                    engine="sglang",
                    engine_args=["--kv-cache-dtype", "fp8_e5m2"],
                    notes="FP8 KV with RadixAttention",
                ),
                Arm(
                    name="sglang-fp16-noapc",
                    engine="sglang",
                    engine_args=["--disable-radix-cache"],
                    notes="control: RadixAttention disabled",
                ),
            ]
        )
        if include_hicache:
            arms.append(
                Arm(
                    name="sglang-fp8-hicache",
                    engine="sglang",
                    engine_args=[
                        "--kv-cache-dtype",
                        "fp8_e5m2",
                        "--enable-hierarchical-cache",
                        "--hicache-size",
                        "4",
                    ],
                    notes="FP8 KV + host-memory hierarchical cache (tiered KV, opt-in)",
                )
            )
    if profile == "smoke":
        keep = {"vllm-fp16-apc", "sglang-fp16-radix"}
        arms = [arm for arm in arms if arm.name in keep]
    if only:
        wanted = set(only)
        arms = [arm for arm in arms if arm.name in wanted]
    return arms


def _system_tokens(context: int, turns: int) -> int:
    if turns <= 1:
        return int(context * 0.90)
    return int(context * 0.45)


def _tool_result_tokens(context: int, turns: int) -> int:
    if turns <= 1:
        return 0
    growth = int(context * 0.90) - _system_tokens(context, turns)
    return max(64, int(growth / turns - 220))


def build_levels(
    contexts: Sequence[int],
    concurrencies: Sequence[int],
    max_tokens: int,
    turns: int = 1,
    decode_probe: bool = True,
) -> List[LevelPlan]:
    mode = "agent" if turns > 1 else "chat"
    levels: List[LevelPlan] = []
    for context in contexts:
        for concurrency in concurrencies:
            levels.append(
                LevelPlan(
                    concurrency=concurrency,
                    turns=turns,
                    system_tokens=_system_tokens(context, turns),
                    tool_result_tokens=_tool_result_tokens(context, turns),
                    max_tokens=max_tokens,
                    mode=mode,
                    label=f"capacity-{context}",
                    context_hint=context,
                    warm_prefix=True,
                )
            )
    if decode_probe:
        for concurrency in (1, 8):
            levels.append(
                LevelPlan(
                    concurrency=concurrency,
                    turns=1,
                    system_tokens=256,
                    tool_result_tokens=0,
                    max_tokens=256,
                    mode="chat",
                    label="decode-512",
                    context_hint=512,
                )
            )
    return levels


def analyze_arm(levels: List[Dict[str, Any]], cfg: Config) -> Dict[str, Any]:
    capacity = [l for l in levels if str(l.get("label", "")).startswith("capacity")]
    decode = [l for l in levels if l.get("label") == "decode-512"]
    by_context: Dict[int, List[Dict[str, Any]]] = {}
    for level in capacity:
        by_context.setdefault(int(level.get("context_hint") or 0), []).append(level)

    context_summary: Dict[str, Dict[str, Any]] = {}
    for context, rows in sorted(by_context.items()):
        rows.sort(key=lambda r: r["concurrency"])
        safe = [r for r in rows if r.get("safe")]
        best = max(safe, key=lambda r: r["concurrency"]) if safe else None
        peak_cache = max(
            [((r.get("engine") or {}).get("cache_usage_peak") or 0) for r in rows] or [0]
        )
        hit_rates = []
        for row in rows:
            hit = row.get("cache_hit_rate_usage")
            if not hit:
                hit = (row.get("engine") or {}).get("apc_hit_rate_metric")
            if hit is not None:
                hit_rates.append(hit)
        context_summary[str(context)] = {
            "concurrency_tested": [r["concurrency"] for r in rows],
            "max_safe_concurrency": best["concurrency"] if best else 0,
            "max_safe_ttft_p99_ms": (best.get("ttft_ms") or {}).get("p99") if best else None,
            "max_safe_output_tok_s": best.get("output_tok_s") if best else None,
            "peak_cache_usage": round(peak_cache, 4),
            "mean_cache_hit_rate": round(sum(hit_rates) / len(hit_rates), 4) if hit_rates else None,
            "actual_prompt_tokens": max([r.get("prompt_tokens") or 0 for r in rows] or [0]),
        }

    decode_summary: Dict[str, Any] = {}
    for row in decode:
        key = f"N{row['concurrency']}"
        decode_summary[key] = {
            "tpot_ms": row.get("tpot_ms"),
            "ttft_ms": row.get("ttft_ms"),
            "output_tok_s": row.get("output_tok_s"),
        }
    return {
        "contexts": context_summary,
        "decode": decode_summary,
    }
