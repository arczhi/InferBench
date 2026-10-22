#!/usr/bin/env python3
"""KV-cache strategy experiment on a single L4.

Measures, for a fixed 24GB card and gpu_memory_utilization=0.9:
  - how many concurrent long-context sessions each KV strategy sustains
  - the TTFT / throughput cost of the extra concurrency
  - the batch-1 vs batch-8 decode penalty of FP8 KV (break-even probe)

Examples:
  python3 -m exp_kv.run --profile smoke --engine vllm --dry-run
  python3 -m exp_kv.run --profile quick --engine all
  python3 -m exp_kv.run --profile full --engine vllm --include-hicache
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import load
from common.engine import DockerEngine
from common.plot import line_chart_svg
from common.report import write_reports
from common.runner import Arm, ExperimentRunner, LevelPlan
from common.util import ROOT, atomic_write_json, ensure_dir

from .arms import PROFILES, analyze_arm, build_arms, build_levels


def _parse_ints(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _estimate(arms: Sequence[Arm], levels: Sequence[LevelPlan]) -> Dict[str, Any]:
    requests = 0
    worst_prompt = 0
    apc_prompt = 0
    output_tokens = 0
    for _arm in arms:
        for level in levels:
            per_session = level.turns * (2 if level.mode == "agent" else 1)
            count = level.concurrency * per_session
            growth = (level.turns - 1) * (level.tool_result_tokens + 220) if level.turns > 1 else 0
            final_prompt = level.system_tokens + growth + 60
            requests += count
            worst_prompt += count * final_prompt
            if level.turns <= 1:
                apc_prompt += level.system_tokens + level.concurrency * (final_prompt * 0.1 + 60)
            else:
                apc_prompt += level.system_tokens + level.concurrency * final_prompt
            output_tokens += count * level.max_tokens
    decode_s = output_tokens / 200.0
    load_s = 120 * len(arms)
    return {
        "requests": requests,
        "estimated_minutes": round((apc_prompt / 4500.0 + decode_s + load_s) / 60.0, 1),
        "estimated_minutes_worst": round((worst_prompt / 4500.0 + decode_s + load_s) / 60.0, 1),
    }


def _series(arms: Sequence[Dict[str, Any]], label: str, value_fn) -> List[Any]:
    out = []
    for arm in arms:
        points = []
        for level in arm.get("levels") or []:
            if level.get("label") != label:
                continue
            value = value_fn(level)
            if value is not None:
                points.append((level.get("concurrency"), value))
        if points:
            out.append((arm.get("name"), points))
    return out


def _charts(summary: Dict[str, Any], cfg, contexts: Sequence[int]) -> List[Any]:
    arms = [a for a in summary.get("arms") or [] if a.get("status") in ("completed", "partial", "failed")]
    charts = []

    for context in contexts:
        label = f"capacity-{context}"
        series = _series(arms, label, lambda level: (level.get("ttft_ms") or {}).get("p99"))
        if series:
            charts.append(
                (
                    f"TTFT p99 vs concurrency, context {context}",
                    line_chart_svg(
                        f"TTFT p99 vs concurrency · context {context}",
                        series,
                        "TTFT p99 (ms)",
                        log_y=True,
                        threshold=cfg.slo.ttft_p99_ms,
                        threshold_label=f"SLO {cfg.slo.ttft_p99_ms:.0f} ms",
                    ),
                )
            )
        series = _series(arms, label, lambda level: level.get("output_tok_s"))
        if series:
            charts.append(
                (
                    f"Output throughput vs concurrency, context {context}",
                    line_chart_svg(
                        f"Output throughput vs concurrency · context {context}",
                        series,
                        "output tokens/s",
                        log_y=False,
                    ),
                )
            )

    capacity_series = []
    for arm in arms:
        analysis = arm.get("analysis") or {}
        points = []
        for context, data in (analysis.get("contexts") or {}).items():
            points.append((context, data.get("max_safe_concurrency")))
        if points:
            capacity_series.append((arm.get("name"), points))
    if capacity_series:
        charts.append(
            (
                "Safe concurrency vs context length",
                line_chart_svg(
                    "Safe concurrency vs context length",
                    capacity_series,
                    "max safe sessions",
                    log_y=False,
                ),
            )
        )

    decode_series = _series(arms, "decode-512", lambda level: (level.get("tpot_ms") or {}).get("p50"))
    if decode_series:
        charts.append(
            (
                "Decode TPOT p50 (batch 1 vs 8)",
                line_chart_svg(
                    "Decode TPOT p50 · short prompt, 256 output tokens",
                    decode_series,
                    "TPOT p50 (ms/token)",
                    log_y=True,
                ),
            )
        )

    if contexts:
        first_label = f"capacity-{contexts[0]}"
        hit_series = _series(
            arms,
            first_label,
            lambda level: (
                round(100 * level["cache_hit_rate_usage"], 2)
                if level.get("cache_hit_rate_usage") is not None
                else None
            ),
        )
        if hit_series:
            charts.append(
                (
                    "Prefix cache hit rate (from usage)",
                    line_chart_svg(
                        f"Prefix cache hit rate · context {contexts[0]}",
                        hit_series,
                        "cached prompt tokens (%)",
                        log_y=False,
                    ),
                )
            )
    return charts


def _decode_comparison(summary: Dict[str, Any]) -> Dict[str, Any]:
    arms = {a.get("name"): a for a in summary.get("arms") or []}
    pairs = [
        ("vllm-fp16-apc", "vllm-fp8-apc"),
        ("sglang-fp16-radix", "sglang-fp8-radix"),
    ]
    out: Dict[str, Any] = {"pairs": []}
    for base_name, test_name in pairs:
        base = arms.get(base_name)
        test = arms.get(test_name)
        if not base or not test:
            continue
        row: Dict[str, Any] = {"baseline": base_name, "candidate": test_name}
        for level in ("N1", "N8"):
            base_tpot = (((base.get("analysis") or {}).get("decode") or {}).get(level) or {}).get("tpot_ms") or {}
            test_tpot = (((test.get("analysis") or {}).get("decode") or {}).get(level) or {}).get("tpot_ms") or {}
            b = base_tpot.get("p50")
            t = test_tpot.get("p50")
            row[f"{level.lower()}_baseline_tpot_p50_ms"] = b
            row[f"{level.lower()}_candidate_tpot_p50_ms"] = t
            row[f"{level.lower()}_slowdown_pct"] = round(100 * (t - b) / b, 1) if b and t else None
        b8 = row.get("n8_baseline_tpot_p50_ms")
        t8 = row.get("n8_candidate_tpot_p50_ms")
        b1 = row.get("n1_baseline_tpot_p50_ms")
        t1 = row.get("n1_candidate_tpot_p50_ms")
        row["break_even_reached"] = bool(b8 and t8 and t8 <= b8)
        row["candidate_slower_at_batch1"] = bool(b1 and t1 and t1 > b1)
        out["pairs"].append(row)
    return out


def _notes(summary: Dict[str, Any], decode: Dict[str, Any]) -> str:
    lines = [
        "- `max_safe_concurrency` = largest N where TTFT p99, TPOT p99 and error rate stay inside the SLO.",
        "- `cache_hit_rate_usage` is computed from response usage (`prompt_tokens_details.cached_tokens`), not from engine counters.",
        "- Decode probe: short prompt, 256 output tokens, N=1 and N=8. Compare FP8 vs FP16 TPOT there to locate the break-even batch.",
        "- SGLang HiCache is the native host-tier KV path; vLLM LMCache/Proxima-style connectors are out of scope for this harness.",
    ]
    for pair in decode.get("pairs") or []:
        lines.append(
            f"- Decode {pair['candidate']} vs {pair['baseline']}: "
            f"N1 slowdown {pair.get('n1_slowdown_pct')}%, "
            f"N8 slowdown {pair.get('n8_slowdown_pct')}%, "
            f"break-even at N8: {pair.get('break_even_reached')}"
        )
    for arm in summary.get("arms") or []:
        if arm.get("skipped_reason"):
            lines.append(f"- `{arm['name']}` skipped: {arm['skipped_reason']}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="L4 KV-cache strategy experiment")
    parser.add_argument("--config", default=None)
    parser.add_argument("--profile", default="quick", choices=sorted(PROFILES))
    parser.add_argument("--engine", default="all", choices=["vllm", "sglang", "all"])
    parser.add_argument("--arms", default=None, help="comma-separated arm names")
    parser.add_argument("--contexts", default=None, help="comma-separated context lengths")
    parser.add_argument("--concurrency", default=None, help="comma-separated concurrency levels")
    parser.add_argument("--turns", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--include-hicache", action="store_true")
    parser.add_argument("--no-decode-probe", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--external-url", default=None)
    parser.add_argument("--level-timeout-s", type=int, default=1800)
    parser.add_argument("--max-wall-s", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-flags", action="store_true")
    args = parser.parse_args(argv)

    cfg = load(args.config)
    if args.model:
        cfg.model = args.model
    if args.max_wall_s:
        cfg.guard.max_wall_s = args.max_wall_s
    elif args.profile == "full" and "EXP_MAX_WALL_S" not in os.environ:
        cfg.guard.max_wall_s = 28800
    profile = PROFILES[args.profile]
    if args.contexts:
        contexts = _parse_ints(args.contexts)
    else:
        contexts = list(profile["contexts"])
    concurrencies = _parse_ints(args.concurrency) if args.concurrency else list(profile["levels"])
    max_tokens = args.max_tokens or int(profile["max_tokens"])
    decode_probe = bool(profile["decode_probe"]) and not args.no_decode_probe
    engines = ["vllm", "sglang"] if args.engine == "all" else [args.engine]
    only = [item.strip() for item in args.arms.split(",")] if args.arms else None

    arms = build_arms(engines, args.profile, args.include_hicache, only)
    if not arms:
        print("[FAIL] no arms selected", file=sys.stderr)
        return 2
    levels = build_levels(contexts, concurrencies, max_tokens, args.turns, decode_probe)
    estimate = _estimate(arms, levels)

    print("=== KV-cache experiment plan ===")
    print(f"profile       : {args.profile}")
    print(f"model         : {cfg.model}")
    print(f"contexts      : {contexts}")
    print(f"concurrency   : {concurrencies}")
    print(f"turns         : {args.turns}  mode={'agent' if args.turns > 1 else 'chat'}")
    print(f"decode probe  : {decode_probe}")
    print(f"arms          : {[a.name for a in arms]}")
    print(
        f"estimate      : {estimate['requests']} requests, "
        f"~{estimate['estimated_minutes']} min with prefix reuse, "
        f"~{estimate['estimated_minutes_worst']} min worst case (excludes model download)"
    )
    print(f"wall guard    : {cfg.guard.max_wall_s}s")
    if estimate["estimated_minutes_worst"] * 60 > cfg.guard.max_wall_s:
        print("[WARN] worst-case estimate exceeds the wall guard; reduce contexts/levels or raise --max-wall-s")

    if args.dry_run:
        ensure_dir(os.path.join(ROOT, "runs", "_dryrun"))
        for arm in arms:
            engine = DockerEngine(
                cfg,
                arm.engine,
                arm.name,
                "dryrun",
                os.path.join(ROOT, "runs", "_dryrun", arm.name),
                arm.engine_args,
            )
            cmd = " ".join(engine.docker_run_cmd())
            print(f"\n[{arm.name}] {arm.notes}\n  {cmd}")
            if args.check_flags:
                missing = engine.check_flags(arm.engine_args)
                print(f"  unsupported flags: {missing or 'none'}")
        print("\nlevels:")
        for level in levels:
            print(
                f"  {level.label:18s} N={level.concurrency:<3d} turns={level.turns} "
                f"system={level.system_tokens} tool={level.tool_result_tokens} "
                f"max_tokens={level.max_tokens} warm_prefix={level.warm_prefix}"
            )
        return 0

    runner = ExperimentRunner(
        cfg=cfg,
        experiment="kv",
        arms=arms,
        levels=levels,
        out_root=args.out_root,
        external_url=args.external_url,
        level_timeout_s=args.level_timeout_s,
        analyzer=analyze_arm,
    )
    result = runner.run()
    decode = _decode_comparison(result)
    atomic_write_json(os.path.join(runner.run_dir, "decode_comparison.json"), decode)
    charts = _charts(result, cfg, contexts)
    paths = write_reports(
        runner.run_dir,
        result,
        charts,
        title="L4 KV-cache strategy experiment",
        notes_md=_notes(result, decode),
    )
    print(f"\nrun dir: {runner.run_dir}")
    print(f"report : {paths['html']}")
    print(f"csv    : {paths['csv']}")
    return 0 if any(a.get("status") in ("completed", "partial") for a in result.get("arms") or []) else 1


if __name__ == "__main__":
    raise SystemExit(main())
