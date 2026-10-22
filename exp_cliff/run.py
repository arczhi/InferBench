#!/usr/bin/env python3
"""Concurrency-cliff experiment on a single L4.

Sweeps session concurrency with an agent-shaped, growing-context workload and
records TTFT / TPOT / throughput, cache occupancy, queue depth and preemptions.
The output is a safe-operating-region table plus the cliff location per arm.

Examples:
  python3 -m exp_cliff.run --profile smoke --engine vllm --dry-run
  python3 -m exp_cliff.run --profile quick --engine all
  python3 -m exp_cliff.run --profile full --engine vllm --context 22700 --turns 12
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import load
from common.engine import DockerEngine
from common.plot import line_chart_svg
from common.report import write_reports
from common.runner import ExperimentRunner
from common.util import ROOT, atomic_write_json, ensure_dir

from .arms import PROFILES, analyze_arm, build_arms, build_levels


def _parse_ints(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _estimate(arms: Sequence[Any], levels: Sequence[Any]) -> Dict[str, Any]:
    requests = 0
    worst_prompt = 0
    apc_prompt = 0
    output_tokens = 0
    for arm in arms:
        for level in levels:
            if arm.max_concurrency is not None and level.concurrency > arm.max_concurrency:
                continue
            per_session = level.turns * (2 if level.mode == "agent" else 1)
            count = level.concurrency * per_session
            growth = (level.turns - 1) * (level.tool_result_tokens + 220) if level.turns > 1 else 0
            final_prompt = level.system_tokens + growth + 60
            requests += count
            worst_prompt += count * final_prompt
            apc_prompt += level.system_tokens + level.concurrency * final_prompt
            output_tokens += count * level.max_tokens
    decode_s = output_tokens / 200.0
    load_s = 120 * len(arms)
    return {
        "requests": requests,
        "estimated_minutes": round((apc_prompt / 4500.0 + decode_s + load_s) / 60.0, 1),
        "estimated_minutes_worst": round((worst_prompt / 4500.0 + decode_s + load_s) / 60.0, 1),
    }


def _series(arms: Sequence[Dict[str, Any]], value_fn, percentile: Optional[str] = None) -> List[Any]:
    out = []
    for arm in arms:
        points = []
        for level in arm.get("levels") or []:
            value = value_fn(level, percentile) if percentile else value_fn(level)
            if value is not None:
                points.append((level.get("concurrency"), value))
        if points:
            out.append((arm.get("name"), points))
    return out


def _charts(summary: Dict[str, Any], cfg) -> List[Any]:
    arms = [a for a in summary.get("arms") or [] if a.get("status") in ("completed", "partial", "failed")]
    charts = []

    ttft_series = _series(arms, lambda level, p: (level.get("ttft_ms") or {}).get(p), "p99")
    if ttft_series:
        charts.append(
            (
                "TTFT p99 vs concurrency",
                line_chart_svg(
                    "TTFT p99 vs concurrency",
                    ttft_series,
                    "TTFT p99 (ms)",
                    log_y=True,
                    threshold=cfg.slo.ttft_p99_ms,
                    threshold_label=f"SLO {cfg.slo.ttft_p99_ms:.0f} ms",
                ),
            )
        )

    throughput_series = _series(arms, lambda level: level.get("output_tok_s"))
    if throughput_series:
        charts.append(
            (
                "Output throughput vs concurrency",
                line_chart_svg(
                    "Output throughput vs concurrency",
                    throughput_series,
                    "output tokens/s",
                    log_y=False,
                ),
            )
        )

    baseline = next((a for a in arms if a.get("name") == "vllm-baseline"), arms[0] if arms else None)
    if baseline:
        pct_series = []
        for percentile in ("p50", "p95", "p99"):
            points = [
                (level.get("concurrency"), (level.get("ttft_ms") or {}).get(percentile))
                for level in baseline.get("levels") or []
                if (level.get("ttft_ms") or {}).get(percentile) is not None
            ]
            if points:
                pct_series.append((f"{baseline['name']} {percentile}", points))
        if pct_series:
            charts.append(
                (
                    "TTFT percentiles (baseline arm)",
                    line_chart_svg(
                        f"TTFT percentiles · {baseline['name']}",
                        pct_series,
                        "TTFT (ms)",
                        log_y=True,
                        threshold=cfg.slo.ttft_p99_ms,
                        threshold_label=f"SLO {cfg.slo.ttft_p99_ms:.0f} ms",
                    ),
                )
            )

        phase_series = []
        for field, label in (
            ("tool_ttft_p99_ms", "tool-call request p99"),
            ("final_ttft_p99_ms", "final-answer request p99"),
            ("agent_roundtrip_p99_ms", "agent round-trip p99 (e2e)"),
        ):
            points = [
                (level.get("concurrency"), (level.get("phases") or {}).get(field))
                for level in baseline.get("levels") or []
                if (level.get("phases") or {}).get(field) is not None
            ]
            if points:
                phase_series.append((label, points))
        if phase_series:
            charts.append(
                (
                    "Agent phase breakdown (baseline arm)",
                    line_chart_svg(
                        f"Agent phase breakdown · {baseline['name']}",
                        phase_series,
                        "latency (ms)",
                        log_y=True,
                    ),
                )
            )

        cache_series = [
            (
                "cache usage peak",
                [
                    (level.get("concurrency"), (level.get("engine") or {}).get("cache_usage_peak"))
                    for level in baseline.get("levels") or []
                    if (level.get("engine") or {}).get("cache_usage_peak") is not None
                ],
            ),
            (
                "queue waiting peak",
                [
                    (level.get("concurrency"), (level.get("engine") or {}).get("waiting_peak"))
                    for level in baseline.get("levels") or []
                    if (level.get("engine") or {}).get("waiting_peak") is not None
                ],
            ),
        ]
        cache_series = [(name, points) for name, points in cache_series if points]
        if cache_series:
            charts.append(
                (
                    "KV cache and queue pressure (baseline arm)",
                    line_chart_svg(
                        f"KV cache and queue pressure · {baseline['name']}",
                        cache_series,
                        "value (fraction / requests)",
                        log_y=False,
                    ),
                )
            )
    return charts


def _comparison(summary: Dict[str, Any]) -> Dict[str, Any]:
    arms = [a for a in summary.get("arms") or [] if a.get("status") in ("completed", "partial")]
    baseline = next((a for a in arms if a.get("name") == "vllm-baseline"), arms[0] if arms else None)
    if not baseline:
        return {}
    base_safe = (baseline.get("cliff") or {}).get("max_safe_concurrency")
    base_cliff = (baseline.get("cliff") or {}).get("cliff_concurrency")
    rows = []
    for arm in summary.get("arms") or []:
        cliff = arm.get("cliff") or {}
        rows.append(
            {
                "arm": arm.get("name"),
                "status": arm.get("status"),
                "max_safe_concurrency": cliff.get("max_safe_concurrency"),
                "cliff_concurrency": cliff.get("cliff_concurrency"),
                "safe_delta_vs_baseline": (cliff.get("max_safe_concurrency") or 0) - (base_safe or 0),
                "cliff_shift_vs_baseline": (cliff.get("cliff_concurrency") or 0) - (base_cliff or 0)
                if cliff.get("cliff_concurrency") and base_cliff
                else None,
                "peak_output_tok_s": (cliff.get("throughput_peak") or {}).get("output_tok_s"),
                "throughput_collapse_at": (cliff.get("throughput_collapse") or {}).get("concurrency"),
                "skipped_reason": arm.get("skipped_reason"),
                "error": arm.get("error"),
            }
        )
    return {"baseline": baseline.get("name"), "rows": rows}


def _notes(summary: Dict[str, Any], comparison: Dict[str, Any]) -> str:
    lines = [
        "- `safe` requires TTFT p99 <= SLO, TPOT p99 <= SLO, and error rate <= SLO. The cliff is the first unsafe concurrency.",
        "- `preemptions_delta > 0` with cache usage near 1.0 is the eviction signature. A cap arm can show queueing (`waiting_peak > 0`) without preemptions.",
        "- Agent mode issues two requests per turn: a forced tool-call request and a final-answer request containing the tool result. The phase chart separates their TTFT.",
        "- A cap arm is expected to trade higher TTFT for stable throughput; compare `throughput_collapse_at` rather than only `cliff_concurrency`.",
    ]
    rows = comparison.get("rows") or []
    if rows:
        lines.append("")
        lines.append("| arm | max safe | cliff | safe Δ | cliff Δ | peak out tok/s |")
        lines.append("|---|---|---|---|---|---|")
        for row in rows:
            lines.append(
                "| {arm} | {max_safe_concurrency} | {cliff_concurrency} | {safe_delta_vs_baseline} | "
                "{cliff_shift_vs_baseline} | {peak_output_tok_s} |".format(**row)
            )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="L4 concurrency-cliff experiment")
    parser.add_argument("--config", default=None)
    parser.add_argument("--profile", default="quick", choices=sorted(PROFILES))
    parser.add_argument("--engine", default="all", choices=["vllm", "sglang", "all"])
    parser.add_argument("--arms", default=None, help="comma-separated arm names")
    parser.add_argument("--context", type=int, default=None)
    parser.add_argument("--levels", default=None, help="comma-separated concurrency levels")
    parser.add_argument("--turns", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--workload", default="agent", choices=["agent", "chat"])
    parser.add_argument("--cap", type=int, default=12, help="admission-cap arm size")
    parser.add_argument("--model", default=None)
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--external-url", default=None)
    parser.add_argument("--level-timeout-s", type=int, default=2400)
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
    context = args.context or int(profile["context"])
    turns = args.turns or int(profile["turns"])
    max_tokens = args.max_tokens or int(profile["max_tokens"])
    concurrencies = _parse_ints(args.levels) if args.levels else list(profile["levels"])
    engines = ["vllm", "sglang"] if args.engine == "all" else [args.engine]
    only = [item.strip() for item in args.arms.split(",")] if args.arms else None

    arms = build_arms(engines, args.profile, args.cap, only)
    if not arms:
        print("[FAIL] no arms selected", file=sys.stderr)
        return 2
    levels = build_levels(context, concurrencies, turns, max_tokens, args.workload)
    estimate = _estimate(arms, levels)

    print("=== concurrency-cliff experiment plan ===")
    print(f"profile     : {args.profile}")
    print(f"model       : {cfg.model}")
    print(f"context     : {context}  turns={turns}  workload={args.workload}")
    print(f"concurrency : {concurrencies}")
    print(f"cap         : {args.cap}")
    print(f"arms        : {[a.name for a in arms]}")
    print(
        f"estimate    : {estimate['requests']} requests, "
        f"~{estimate['estimated_minutes']} min with prefix reuse, "
        f"~{estimate['estimated_minutes_worst']} min worst case (excludes model download)"
    )
    print(f"wall guard  : {cfg.guard.max_wall_s}s")
    if max(concurrencies) <= args.cap and any("cap" in arm.name for arm in arms):
        print(f"[WARN] max concurrency {max(concurrencies)} <= cap {args.cap}; the cap arm will not queue")
    if estimate["estimated_minutes_worst"] * 60 > cfg.guard.max_wall_s:
        print("[WARN] worst-case estimate exceeds the wall guard; reduce levels or raise --max-wall-s")

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
            print(f"\n[{arm.name}] {arm.notes}\n  {' '.join(engine.docker_run_cmd())}")
            if args.check_flags:
                print(f"  unsupported flags: {engine.check_flags(arm.engine_args) or 'none'}")
        print("\nlevels:")
        for level in levels:
            print(
                f"  N={level.concurrency:<3d} turns={level.turns} mode={level.mode} "
                f"system={level.system_tokens} tool_result={level.tool_result_tokens} "
                f"max_tokens={level.max_tokens} approx_context={level.context_hint}"
            )
        return 0

    runner = ExperimentRunner(
        cfg=cfg,
        experiment="cliff",
        arms=arms,
        levels=levels,
        out_root=args.out_root,
        external_url=args.external_url,
        level_timeout_s=args.level_timeout_s,
        analyzer=analyze_arm,
    )
    result = runner.run()
    comparison = _comparison(result)
    atomic_write_json(os.path.join(runner.run_dir, "comparison.json"), comparison)
    charts = _charts(result, cfg)
    paths = write_reports(
        runner.run_dir,
        result,
        charts,
        title="L4 concurrency-cliff experiment",
        notes_md=_notes(result, comparison),
    )
    print(f"\nrun dir: {runner.run_dir}")
    print(f"report : {paths['html']}")
    print(f"csv    : {paths['csv']}")
    return 0 if any(a.get("status") in ("completed", "partial") for a in result.get("arms") or []) else 1


if __name__ == "__main__":
    raise SystemExit(main())
