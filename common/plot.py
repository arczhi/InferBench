"""Tiny dependency-free SVG line charts for experiment reports."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape

PALETTE = [
    "#2563eb",
    "#dc2626",
    "#059669",
    "#d97706",
    "#7c3aed",
    "#0891b2",
    "#be185d",
    "#4d7c0f",
    "#9333ea",
    "#b45309",
]

WIDTH = 960
HEIGHT = 460
LEFT = 78
RIGHT = 24
TOP = 54
BOTTOM = 74


def _nice_linear(min_v: float, max_v: float, ticks: int = 5) -> Tuple[float, float, List[float]]:
    if max_v <= min_v:
        max_v = min_v + 1.0
    span = max_v - min_v
    raw = span / max(1, ticks - 1)
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    norm = raw / mag
    step = (1 if norm <= 1 else 2 if norm <= 2 else 5 if norm <= 5 else 10) * mag
    lo = math.floor(min_v / step) * step
    hi = math.ceil(max_v / step) * step
    values = []
    value = lo
    while value <= hi + step * 0.01:
        values.append(round(value, 6))
        value += step
    return lo, hi, values


def _nice_log(min_v: float, max_v: float) -> Tuple[float, float, List[float]]:
    min_v = max(min_v, 1e-6)
    lo_exp = math.floor(math.log10(min_v))
    hi_exp = math.ceil(math.log10(max_v))
    lo = 10 ** lo_exp
    hi = 10 ** hi_exp
    if hi <= lo:
        hi = lo * 10
    ticks = []
    for exp in range(lo_exp, hi_exp + 1):
        base = 10 ** exp
        for mult in (1, 2, 5):
            value = base * mult
            if lo <= value <= hi:
                ticks.append(value)
    if len(ticks) < 2:
        ticks = [lo, hi]
    return lo, hi, ticks


def _fmt(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:.0f}"
    if abs(value) >= 1:
        return f"{value:.1f}"
    return f"{value:.3g}"


def line_chart_svg(
    title: str,
    series: Sequence[Tuple[str, Sequence[Tuple[Any, Optional[float]]]]],
    y_label: str,
    log_y: bool = False,
    threshold: Optional[float] = None,
    threshold_label: str = "",
    annotations: Optional[Dict[str, str]] = None,
) -> str:
    xs: List[str] = []
    for _name, points in series:
        for x, _y in points:
            key = str(x)
            if key not in xs:
                xs.append(key)
    try:
        xs.sort(key=lambda value: float(value))
    except ValueError:
        xs.sort()

    numeric: List[float] = []
    for _name, points in series:
        for _x, y in points:
            if y is not None and (y > 0 or not log_y):
                numeric.append(float(y))
    if threshold is not None and threshold > 0:
        numeric.append(float(threshold))
    if not numeric:
        return _empty_svg(title, "no data")
    min_y = min(numeric)
    max_y = max(numeric)
    if log_y:
        lo, hi, ticks = _nice_log(min_y, max_y)
        scale = lambda value: TOP + (HEIGHT - TOP - BOTTOM) * (
            1 - (math.log10(max(value, lo)) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
        )
    else:
        pad = (max_y - min_y) * 0.08 or 1.0
        lo, hi, ticks = _nice_linear(min_y - pad, max_y + pad)
        scale = lambda value: TOP + (HEIGHT - TOP - BOTTOM) * (1 - (value - lo) / (hi - lo))

    plot_w = WIDTH - LEFT - RIGHT
    plot_h = HEIGHT - TOP - BOTTOM
    x_positions = {
        key: LEFT + (plot_w * i / max(1, len(xs) - 1)) if len(xs) > 1 else LEFT + plot_w / 2
        for i, key in enumerate(xs)
    }

    parts: List[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" font-family="Inter, -apple-system, Segoe UI, sans-serif">'
    )
    parts.append(f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>')
    parts.append(
        f'<text x="{LEFT}" y="26" font-size="16" font-weight="600" fill="#111827">{escape(title)}</text>'
    )
    parts.append(
        f'<text x="{LEFT}" y="{HEIGHT - 22}" font-size="12" fill="#6b7280">concurrency (sessions)</text>'
    )
    parts.append(
        f'<text x="18" y="{TOP + plot_h / 2}" font-size="12" fill="#6b7280" '
        f'transform="rotate(-90 18 {TOP + plot_h / 2})">{escape(y_label)}</text>'
    )
    for tick in ticks:
        y = scale(tick)
        if y < TOP - 1 or y > HEIGHT - BOTTOM + 1:
            continue
        parts.append(f'<line x1="{LEFT}" y1="{y:.1f}" x2="{WIDTH - RIGHT}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(
            f'<text x="{LEFT - 10}" y="{y + 4:.1f}" font-size="11" fill="#6b7280" text-anchor="end">{_fmt(tick)}</text>'
        )
    for key in xs:
        x = x_positions[key]
        parts.append(f'<line x1="{x:.1f}" y1="{TOP}" x2="{x:.1f}" y2="{HEIGHT - BOTTOM}" stroke="#f3f4f6" stroke-width="1"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{HEIGHT - BOTTOM + 20}" font-size="11" fill="#374151" text-anchor="middle">{escape(key)}</text>'
        )
    parts.append(f'<line x1="{LEFT}" y1="{HEIGHT - BOTTOM}" x2="{WIDTH - RIGHT}" y2="{HEIGHT - BOTTOM}" stroke="#9ca3af"/>')
    parts.append(f'<line x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{HEIGHT - BOTTOM}" stroke="#9ca3af"/>')

    if threshold is not None and threshold > 0 and lo <= threshold <= hi:
        y = scale(threshold)
        label = threshold_label or f"threshold {_fmt(threshold)}"
        parts.append(
            f'<line x1="{LEFT}" y1="{y:.1f}" x2="{WIDTH - RIGHT}" y2="{y:.1f}" stroke="#ef4444" '
            f'stroke-width="1.5" stroke-dasharray="6 5"/>'
        )
        parts.append(
            f'<text x="{WIDTH - RIGHT - 4}" y="{y - 6:.1f}" font-size="11" fill="#ef4444" text-anchor="end">{escape(label)}</text>'
        )

    legend_y = TOP - 18
    legend_x = LEFT
    for index, (name, points) in enumerate(series):
        color = PALETTE[index % len(PALETTE)]
        coords = []
        for x, y in points:
            key = str(x)
            if key not in x_positions or y is None:
                continue
            coords.append((x_positions[key], scale(float(y)), y))
        if coords:
            path = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in coords)
            parts.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2.4"/>')
            for x, y, value in coords:
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" stroke="#fff" stroke-width="1.5"/>')
        legend_text = escape(str(name))
        parts.append(
            f'<rect x="{legend_x}" y="{legend_y - 9}" width="10" height="10" rx="2" fill="{color}"/>'
            f'<text x="{legend_x + 15}" y="{legend_y}" font-size="11" fill="#374151">{legend_text}</text>'
        )
        legend_x += 18 + 7 * len(str(name)) + 18
        if legend_x > WIDTH - RIGHT - 120:
            legend_x = LEFT
            legend_y += 16

    if annotations:
        for key, text in annotations.items():
            if key in x_positions:
                x = x_positions[key]
                parts.append(
                    f'<text x="{x:.1f}" y="{TOP + 14}" font-size="11" fill="#b91c1c" text-anchor="middle">{escape(text)}</text>'
                )
    parts.append("</svg>")
    return "".join(parts)


def _empty_svg(title: str, message: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}">'
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#fff"/>'
        f'<text x="{LEFT}" y="30" font-size="16" fill="#111827">{escape(title)}</text>'
        f'<text x="{LEFT}" y="{HEIGHT / 2}" font-size="13" fill="#6b7280">{escape(message)}</text></svg>'
    )
