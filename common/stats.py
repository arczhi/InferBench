from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence


def percentile(values: Sequence[float], p: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(float(ordered[0]), 3)
    rank = (len(ordered) - 1) * (p / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    value = ordered[low] * (1 - frac) + ordered[high] * frac
    return round(float(value), 3)


def summarize(values: Iterable[float]) -> Dict[str, Any]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"count": 0}
    mean = sum(clean) / len(clean)
    variance = sum((v - mean) ** 2 for v in clean) / len(clean)
    return {
        "count": len(clean),
        "mean": round(mean, 3),
        "p50": percentile(clean, 50),
        "p90": percentile(clean, 90),
        "p95": percentile(clean, 95),
        "p99": percentile(clean, 99),
        "min": round(min(clean), 3),
        "max": round(max(clean), 3),
        "std": round(variance ** 0.5, 3),
    }


def safe_div(num: float, den: float) -> Optional[float]:
    if not den:
        return None
    return num / den


def table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    if not rows:
        return "(no rows)"
    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            widths[col] = max(widths[col], len(_cell(row.get(col))))
    header = " | ".join(col.ljust(widths[col]) for col in columns)
    separator = "-+-".join("-" * widths[col] for col in columns)
    lines = [header, separator]
    for row in rows:
        lines.append(" | ".join(_cell(row.get(col)).ljust(widths[col]) for col in columns))
    return "\n".join(lines)


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if value == int(value):
            return str(int(value))
        return f"{value:.2f}"
    return str(value)
