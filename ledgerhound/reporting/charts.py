"""
ledgerhound.reporting.charts
=========================

Matplotlib chart factories that return base64-encoded PNG strings ready
to drop into an ``<img src="data:image/png;base64,...">`` element.

All charts use ``matplotlib.use("Agg")`` so they work in headless
environments (CI, server-side rendering, etc.).
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)


# -- House style ------------------------------------------------------------


_BG     = "#ffffff"
_TEXT   = "#222222"
_GRID   = "#e6e6e6"
_GREEN  = "#2ca02c"
_YELLOW = "#f0a500"
_RED    = "#d62728"
_BLUE   = "#1f77b4"
_GREY   = "#888888"

_SEV_COLOURS = {
    "PASS":     _GREEN,
    "WARNING":  _YELLOW,
    "WARN":     _YELLOW,
    "FAIL":     _RED,
    "ERROR":    _RED,
    "SKIPPED":  _GREY,
    "INFO":     _BLUE,
    "ALERT":    _RED,
    "NONE":     _GREEN,
    "OK":       _GREEN,
}


def _png_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110,
                facecolor=_BG, edgecolor="none")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# -- Charts -----------------------------------------------------------------


def chart_playbook_steps(steps: list[dict]) -> str:
    """Horizontal bar chart of playbook step elapsed times, coloured by status."""
    if not steps:
        return ""
    names    = [s.get("step_name", "?") for s in steps]
    times_ms = [float(s.get("elapsed_ms", 0)) for s in steps]
    colours  = [_SEV_COLOURS.get(s.get("status", "OK"), _GREY) for s in steps]

    fig, ax = plt.subplots(figsize=(8, 0.5 + 0.4 * len(steps)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, times_ms, color=colours, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("elapsed (ms)")
    ax.set_title("Playbook step timeline", color=_TEXT, loc="left", fontsize=11)
    ax.grid(axis="x", color=_GRID, linestyle="-", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _png_b64(fig)


def chart_scorecard_radar(components: list[dict]) -> str:
    """Radar chart of 5 scorecard components."""
    if not components:
        return ""
    names  = [c.get("name", "?") for c in components]
    scores = [float(c.get("score", 0)) for c in components]
    angles = np.linspace(0, 2 * np.pi, len(names), endpoint=False).tolist()
    scores_loop = scores + scores[:1]
    angles_loop = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], color=_GREY, fontsize=8)
    ax.set_xticks(angles)
    ax.set_xticklabels(names, color=_TEXT, fontsize=10)
    ax.plot(angles_loop, scores_loop, color=_BLUE, linewidth=2)
    ax.fill(angles_loop, scores_loop, color=_BLUE, alpha=0.20)
    # Marker points
    for ang, sc in zip(angles, scores):
        c = _GREEN if sc >= 0.7 else (_YELLOW if sc >= 0.5 else _RED)
        ax.scatter([ang], [sc], color=c, s=50, zorder=10, edgecolor="white", linewidth=1.0)
    ax.set_title("Scorecard components", color=_TEXT, fontsize=11, pad=18)
    return _png_b64(fig)


def chart_scorecard_bars(components: list[dict]) -> str:
    """Vertical bars showing weighted contribution of each component."""
    if not components:
        return ""
    names    = [c.get("name", "?") for c in components]
    scores   = [float(c.get("score", 0)) for c in components]
    weights  = [float(c.get("weight", 0)) for c in components]
    weighted = [s * w for s, w in zip(scores, weights)]
    colours  = [_GREEN if s >= 0.7 else (_YELLOW if s >= 0.5 else _RED) for s in scores]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(names))
    width = 0.36
    ax.bar(x - width/2, scores,   width, label="raw score", color=colours, edgecolor="white")
    ax.bar(x + width/2, weighted, width, label="weighted",  color=_BLUE,   edgecolor="white", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", color=_TEXT, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.grid(axis="y", color=_GRID, linestyle="-", linewidth=0.5)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Component scores (raw vs weighted)", color=_TEXT,
                 loc="left", fontsize=11)
    return _png_b64(fig)


def chart_monitor_timeline(runs: list[dict]) -> str:
    """Timeline of monitor cycles coloured by overall_alert severity."""
    if not runs:
        return ""
    periods    = [r.get("period", "?") for r in runs]
    cycle_ids  = [r.get("cycle_id", "")[-8:] for r in runs]
    severities = [r.get("overall_alert", "NONE") for r in runs]
    decay      = [int(r.get("n_signal_decay_events", 0)) for r in runs]
    evo        = [int(r.get("n_evolution_events", 0)) for r in runs]

    fig, ax = plt.subplots(figsize=(9, 3.5))
    x = np.arange(len(runs))
    width = 0.38
    ax.bar(x - width/2, decay, width, label="signal-decay events", color=_RED,
           edgecolor="white", alpha=0.85)
    ax.bar(x + width/2, evo,   width, label="evolution events",   color=_BLUE,
           edgecolor="white", alpha=0.85)
    # Severity dots
    for i, sev in enumerate(severities):
        ax.scatter([i], [max(decay[i], evo[i]) + 0.4],
                   color=_SEV_COLOURS.get(sev, _GREY), s=80, zorder=10,
                   edgecolor="white", linewidth=1.0)

    labels = [f"{p}\n{c}" for p, c in zip(periods, cycle_ids)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, color=_TEXT)
    ax.set_ylabel("events")
    ax.grid(axis="y", color=_GRID, linestyle="-", linewidth=0.5)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Monitor cycle timeline", color=_TEXT, loc="left", fontsize=11)
    return _png_b64(fig)


def chart_eval_per_pattern(per_pattern: list[dict]) -> str:
    """Grouped bar chart of precision/recall/F1 per motif pattern."""
    if not per_pattern:
        return ""
    names  = [p.get("pattern_name", "?") for p in per_pattern]
    prec   = [float(p.get("precision", 0)) for p in per_pattern]
    rec    = [float(p.get("recall", 0))    for p in per_pattern]
    f1     = [float(p.get("f1", 0))        for p in per_pattern]

    fig, ax = plt.subplots(figsize=(8.5, 4))
    x = np.arange(len(names))
    width = 0.27
    ax.bar(x - width, prec, width, label="precision", color=_BLUE,   edgecolor="white")
    ax.bar(x,         rec,  width, label="recall",    color=_GREEN,  edgecolor="white")
    ax.bar(x + width, f1,   width, label="F1",        color=_YELLOW, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", color=_TEXT, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("metric")
    ax.grid(axis="y", color=_GRID, linestyle="-", linewidth=0.5)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Per-pattern precision / recall / F1", color=_TEXT,
                 loc="left", fontsize=11)
    return _png_b64(fig)


def chart_eval_summary(metrics: dict) -> str:
    """Single bar chart for top-line account-level metrics."""
    am = metrics.get("account_metrics") or {}
    if not am:
        return ""
    keys   = ["precision", "recall", "f1", "accuracy", "roc_auc", "pr_auc"]
    values = []
    for k in keys:
        v = am.get(k)
        values.append(float(v) if v is not None else 0.0)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(keys))
    colours = [_GREEN if v >= 0.85 else (_YELLOW if v >= 0.7 else _RED) for v in values]
    ax.bar(x, values, color=colours, edgecolor="white")
    for i, v in enumerate(values):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", va="bottom",
                fontsize=9, color=_TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, color=_TEXT, fontsize=10)
    ax.set_ylim(0, 1.10)
    ax.grid(axis="y", color=_GRID, linestyle="-", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Account-level classification", color=_TEXT,
                 loc="left", fontsize=11)
    return _png_b64(fig)
