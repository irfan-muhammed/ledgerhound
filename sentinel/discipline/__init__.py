"""
sentinel.discipline
===================

Formal epistemic discipline primitives for Sentinel.

This package contains the formal contribution of the Sentinel system
that distinguishes it from generic LLM-EDA agents (Code Interpreter,
DataInterpreter, Pandas-AI, DSAgent, TaskWeaver, etc.):

  **Sample-frame discipline as a first-class agent constraint.**

Modules
-------
- ``sample_frame`` -- the SampleFrame enum, TaggedFrame dataclass,
                     SampleFrameAlgebra rules, and SampleFrameValidator
                     for checking whether a claim is consistent with
                     its declared sample frame.
"""

from sentinel.discipline.sample_frame import (
    OperationKind,
    SampleFrame,
    SampleFrameAlgebra,
    SampleFrameValidator,
    SampleFrameViolation,
    TaggedFrame,
    operation_table,
)

__all__ = [
    "SampleFrame",
    "TaggedFrame",
    "SampleFrameViolation",
    "SampleFrameAlgebra",
    "SampleFrameValidator",
    "OperationKind",
    "operation_table",
]
