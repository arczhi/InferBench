from __future__ import annotations

import csv
import html
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .stats import table


def _arm_row(arm: Dict[str, Any]) -> Dict[str, Any]:
    cliff = arm.get("cliff") or {}
    peak = (cliff.get("throughput_peak") or {}) if cliff else {}
    return {
        "arm": arm.get("name"),
        "engine": arm.get("engine"),
        "status": arm.get("status"),
        "levels": len(arm.get("levels") or []),
        "max_safe_n": cliff.get("max_safe_concurrency"),
        "cliff_n": cliff.get("cliff_concurrency"),
        "peak_tok_s": peak.get("output_tok_s"),
        "peak_at_n": peak.get("concurrency"),
        "note": arm.get("skipped_reason") or arm.get("error") or arm.get("notes") or "",
    }


def _level_rows(arm: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for level in arm.get("levels") or []:
        engine = level.get("engine") or {}
        rows.append(
            {
                "N": level.get("concurrency"),
                "req": level.get("requests"),
                "ok": level.get("requests_ok"),
                "err%": round(100 * (level.get("error_rate") or 0), 1),
                "ttft p50": (level.get("ttft_ms") or {}).get("p50"),
                "ttft p99": (level.get("ttft_ms") or {}).get("p99"),
                "tpot p99": (level.get("tpot_ms") or {}).get("p99"),
                "out tok/s": level.get("output_tok_s"),
                "prefill tok/s": level.get("prefill_tok_s"),
                "cache%": round(100 * engine["cache_usage_peak"], 1) if engine.get("cache_usage_peak") is not None else None,
                "wait": engine.get("waiting_peak"),
                "preempt": engine.get("preemptions_delta"),
                "apc hit%": round(
                    100
                    * (
                        level.get("cache_hit_rate_usage")
                        or (engine.get("apc_hit_rate_metric") or 0)
                    ),
                    1,
                )
                if (level.get("cache_hit_rate_usage") is not None or engine.get("apc_hit_rate_metric") is not None)
                else None,
                "safe": "yes" if level.get("safe") else "no",
            }
        )
    return rows


def _md_table(rows: List[Dict[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return "_(none)_"
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col)
            cells.append("" if value is None else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_csv(run_dir: str, summary: Dict[str, Any]) -> Optional[str]:
    path = os.path.join(run_dir, "levels.csv")
    rows = []
    for arm in summary.get("arms") or []:
        for level in arm.get("levels") or []:
            engine = level.get("engine") or {}
            phases = level.get("phases") or {}
            rows.append(
                {
                    "run_id": summary.get("run_id"),
                    "arm": arm.get("name"),
                    "engine": arm.get("engine"),
                    "workload": level.get("workload"),
                    "concurrency": level.get("concurrency"),
                    "requests": level.get("requests"),
                    "requests_ok": level.get("requests_ok"),
                    "error_rate": level.get("error_rate"),
                    "wall_s": level.get("wall_s"),
                    "ttft_p50_ms": (level.get("ttft_ms") or {}).get("p50"),
                    "ttft_p95_ms": (level.get("ttft_ms") or {}).get("p95"),
                    "ttft_p99_ms": (level.get("ttft_ms") or {}).get("p99"),
                    "tpot_p50_ms": (level.get("tpot_ms") or {}).get("p50"),
                    "tpot_p99_ms": (level.get("tpot_ms") or {}).get("p99"),
                    "e2e_p99_ms": (level.get("e2e_ms") or {}).get("p99"),
                    "prompt_tokens": level.get("prompt_tokens"),
                    "cached_tokens": level.get("cached_tokens"),
                    "completion_tokens": level.get("completion_tokens"),
                    "cache_hit_rate_usage": level.get("cache_hit_rate_usage"),
                    "output_tok_s": level.get("output_tok_s"),
                    "prefill_tok_s": level.get("prefill_tok_s"),
                    "cache_usage_peak": engine.get("cache_usage_peak"),
                    "waiting_peak": engine.get("waiting_peak"),
                    "preemptions_delta": engine.get("preemptions_delta"),
                    "apc_hit_rate_metric": engine.get("apc_hit_rate_metric"),
                    "tool_ttft_p99_ms": ((phases.get("tool") or {}).get("ttft_ms") or {}).get("p99"),
                    "final_ttft_p99_ms": ((phases.get("final") or {}).get("ttft_ms") or {}).get("p99"),
                    "agent_roundtrip_p99_ms": (phases.get("agent_roundtrip_e2e_ms") or {}).get("p99"),
                    "safe": level.get("safe"),
                }
            )
    if not rows:
        return None
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_reports(
    run_dir: str,
    summary: Dict[str, Any],
    charts: Sequence[Tuple[str, str]],
    title: str,
    notes_md: str = "",
) -> Dict[str, str]:
    os.makedirs(run_dir, exist_ok=True)
    arms = summary.get("arms") or []
    overview_cols = ["arm", "engine", "status", "levels", "max_safe_n", "cliff_n", "peak_tok_s", "peak_at_n", "note"]
    overview = [_arm_row(arm) for arm in arms]

    md_lines = [
        f"# {title}",
        "",
        f"- run: `{summary.get('run_id')}`",
        f"- started: {summary.get('started_at')}  finished: {summary.get('finished_at')}  duration: {summary.get('duration_s')}s",
        f"- aborted: {summary.get('aborted')}  reason: {summary.get('abort_reason')}",
        "",
        "## Arms",
        "",
        _md_table(overview, overview_cols),
        "",
    ]
    for arm in arms:
        md_lines.append(f"### {arm.get('name')} ({arm.get('engine')})")
        if arm.get("skipped_reason"):
            md_lines.append(f"skipped: {arm['skipped_reason']}")
        if arm.get("error"):
            md_lines.append(f"error: {arm['error']}")
        cliff = arm.get("cliff") or {}
        if cliff:
            md_lines.append(
                f"max safe N={cliff.get('max_safe_concurrency')}  cliff N={cliff.get('cliff_concurrency')}  "
                f"threshold={cliff.get('unsafe_threshold_ms')}ms"
            )
        md_lines.append("")
        md_lines.append(
            _md_table(
                _level_rows(arm),
                ["N", "req", "ok", "err%", "ttft p50", "ttft p99", "tpot p99", "out tok/s", "prefill tok/s", "cache%", "wait", "preempt", "apc hit%", "safe"],
            )
        )
        md_lines.append("")
    if notes_md:
        md_lines.append("## Notes")
        md_lines.append(notes_md)

    with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md_lines))

    overview_html = _md_table(overview, overview_cols)
    body = [
        f"<h1>{html.escape(title)}</h1>",
        f'<p class="meta">run <code>{html.escape(str(summary.get("run_id")))}</code> · '
        f'{html.escape(str(summary.get("started_at")))} → {html.escape(str(summary.get("finished_at")))} · '
        f'{summary.get("duration_s")}s · aborted={summary.get("aborted")}</p>',
    ]
    if summary.get("aborted"):
        body.append(f'<p class="alert">Aborted by guard: {html.escape(str(summary.get("abort_reason")))}</p>')
    body.append("<h2>Arms</h2>")
    body.append(_html_table(overview, overview_cols))
    if charts:
        body.append("<h2>Curves</h2>")
        for chart_title, svg in charts:
            body.append(f'<div class="chart"><h3>{html.escape(chart_title)}</h3>{svg}</div>')
    for arm in arms:
        body.append(f'<h2>{html.escape(str(arm.get("name")))} <span class="dim">({html.escape(str(arm.get("engine")))})</span></h2>')
        if arm.get("skipped_reason"):
            body.append(f'<p class="alert">skipped: {html.escape(str(arm["skipped_reason"]))}</p>')
        if arm.get("error"):
            body.append(f'<p class="alert">error: {html.escape(str(arm["error"]))}</p>')
        body.append(
            _html_table(
                _level_rows(arm),
                ["N", "req", "ok", "err%", "ttft p50", "ttft p99", "tpot p99", "out tok/s", "prefill tok/s", "cache%", "wait", "preempt", "apc hit%", "safe"],
            )
        )
    if notes_md:
        body.append("<h2>Notes</h2>")
        body.append("<pre>" + html.escape(notes_md) + "</pre>")
    body.append("<details><summary>Run config</summary><pre>")
    body.append(html.escape(json.dumps(summary.get("config") or {}, indent=2, default=str)))
    body.append("</pre></details>")

    html_doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
:root {{ color-scheme: light; }}
body {{ margin: 0; padding: 32px 40px 80px; font: 14px/1.5 Inter, -apple-system, "Segoe UI", sans-serif;
       color: #111827; background: #fafafa; max-width: 1180px; }}
h1 {{ font-size: 26px; margin: 0 0 8px; letter-spacing: -0.02em; }}
h2 {{ font-size: 18px; margin: 36px 0 12px; }}
h3 {{ font-size: 14px; margin: 18px 0 6px; }}
.meta, .dim {{ color: #6b7280; font-weight: 400; }}
.alert {{ background: #fef2f2; border: 1px solid #fecaca; color: #991b1b; padding: 8px 12px; border-radius: 8px; }}
table {{ border-collapse: collapse; width: 100%; background: #fff; border: 1px solid #e5e7eb; font-size: 12.5px; }}
th {{ text-align: left; background: #f9fafb; font-weight: 600; }}
th, td {{ padding: 7px 9px; border-bottom: 1px solid #f0f0f0; white-space: nowrap; }}
code {{ background: #f3f4f6; padding: 1px 5px; border-radius: 4px; }}
.chart {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 14px; margin: 16px 0; overflow-x: auto; }}
pre {{ background: #111827; color: #e5e7eb; padding: 16px; border-radius: 10px; overflow-x: auto; font-size: 12px; }}
details {{ margin-top: 24px; }}
</style></head><body>
{''.join(body)}
</body></html>"""
    with open(os.path.join(run_dir, "report.html"), "w", encoding="utf-8") as fh:
        fh.write(html_doc)

    csv_path = write_csv(run_dir, summary)
    return {
        "html": os.path.join(run_dir, "report.html"),
        "md": os.path.join(run_dir, "report.md"),
        "csv": csv_path or "",
    }


def _html_table(rows: List[Dict[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return "<p>(none)</p>"
    parts = ["<table><thead><tr>"]
    for col in columns:
        parts.append(f"<th>{html.escape(str(col))}</th>")
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for col in columns:
            value = row.get(col)
            parts.append(f"<td>{'' if value is None else html.escape(str(value))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)
