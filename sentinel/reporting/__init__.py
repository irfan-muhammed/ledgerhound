"""
sentinel.reporting
==================

Render Sentinel artifacts (playbook reports, scorecards, monitor cycles,
evaluation reports) as self-contained HTML -- or PDF when weasyprint is
installed.

Design rules
------------
- **Self-contained HTML.**  Charts are matplotlib PNGs base64-embedded
  into the template.  No external scripts, no CDN dependencies, no
  network needed to view.  Open the file in a browser, share by email.
- **Auto-detect input type.**  ``HtmlReportRenderer.render_auto(path)``
  inspects the JSON and picks the right template (playbook / scorecard /
  monitor / eval).
- **Pure renderer.**  No I/O on input -- the renderer takes already-loaded
  dicts.  The CLI handles file reads.

Public surface
--------------
- ``HtmlReportRenderer``
- ``render_playbook_html``
- ``render_scorecard_html``
- ``render_monitor_history_html``
- ``render_evaluation_html``
"""

# NOTE: this package is half-built -- see docs/HANDOFF_C2_C3.md
# charts.py is complete; html_renderer.py + templates/ + CLI still pending.
# The exports below are commented out until html_renderer.py exists, so
# `import sentinel.reporting` doesn't break.

# from sentinel.reporting.html_renderer import (
#     HtmlReportRenderer,
#     render_playbook_html,
#     render_scorecard_html,
#     render_monitor_history_html,
#     render_evaluation_html,
# )

from sentinel.reporting.charts import (
    chart_playbook_steps,
    chart_scorecard_radar,
    chart_scorecard_bars,
    chart_monitor_timeline,
    chart_eval_per_pattern,
    chart_eval_summary,
)

__all__ = [
    "chart_playbook_steps",
    "chart_scorecard_radar",
    "chart_scorecard_bars",
    "chart_monitor_timeline",
    "chart_eval_per_pattern",
    "chart_eval_summary",
]
