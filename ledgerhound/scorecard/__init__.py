"""
ledgerhound.scorecard
==================

The Ledgerhound EDA Scorecard -- a single artifact that answers:

  *Is this EDA done?*  (completeness)
  *Is it correct?*     (discipline + corroboration)
  *How thorough?*      (data quality + process)

The scorecard rolls up all the metrics Ledgerhound produces -- findings,
coverage, validation, motif/community/fingerprint outputs, knowledge
deposits -- into one composite score with five named components.

Public surface
--------------
``Scorecard.from_workspace(workspace, dataset_name)`` is the high-level
entry point.  It reconstructs all the inputs from the workspace's reports
and journals, scores each component, and returns a fully populated
``Scorecard`` ready to ``.display()`` or ``.save()``.

Components (each scored 0..1, then weighted into the overall score):
  - completeness  (0.25)  -- mandatory steps + column / stratum coverage
  - discipline    (0.30)  -- sample-frame validation rate
  - corroboration (0.15)  -- cross-tool agreement on alerts
  - data_quality  (0.15)  -- PSI / consistency / signal-grade
  - process       (0.15)  -- tool count, time, KB consultation
"""

from ledgerhound.scorecard.scorecard import Scorecard
from ledgerhound.scorecard.components import (
    ComponentScore,
    completeness_score,
    discipline_score,
    corroboration_score,
    data_quality_score,
    process_score,
)
from ledgerhound.scorecard.corroboration import (
    CorroborationMatrix,
    TooledAlert,
    build_corroboration_matrix,
)

__all__ = [
    "Scorecard",
    "ComponentScore",
    "completeness_score",
    "discipline_score",
    "corroboration_score",
    "data_quality_score",
    "process_score",
    "CorroborationMatrix",
    "TooledAlert",
    "build_corroboration_matrix",
]
