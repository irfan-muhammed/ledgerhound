"""
sentinel.methodology
======================

Declared, analyst-chosen analysis methodologies: an ordered sequence of
phases, each with its own data scope, allowed grains, and a checklist of
concrete EDA steps.

The tool ships templates; it does not impose one. Per senior feedback
(2026-08-07), the fraud team runs several legitimate methodologies and
analysts differ by experience -- nothing is right or wrong until the end
goal, optimal fraud-model precision, is reached. Transaction-level analysis
is a necessary part of these methods, not a defect.

See phases.py for the model and templates/ for the shipped sequences.
"""

from sentinel.methodology.phases import (  # noqa: F401
    GRAIN_ACCOUNT, GRAIN_ANY, GRAIN_TXN,
    SCOPE_BOTH, SCOPE_FRAUD, SCOPE_NONFRAUD,
    MODE_FULL, MODE_EDA_ONLY, MODE_FE_ONLY, MODE_EDA_FE_ONLY,
    Methodology, Phase, Step,
    available_templates, available_modes, from_dict, load_template,
)
