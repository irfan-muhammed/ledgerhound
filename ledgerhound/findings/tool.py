"""
ledgerhound.findings.tool
======================

Agent-callable wrappers over the Findings log.

Two tools are exposed to the ReAct loop:

* ``record_finding(...)`` -- append one structured Finding to the log.
* ``list_findings(limit=10)`` -- peek at recently recorded findings (this
  session or earlier).

Both are bound to a specific ``FindingsLog`` and an optional
``DatasetMemory``.  The memory is used for **sample-frame discipline
checks** (P1.E): claims tagged ``population_estimate`` are flagged as
suspicious when the dataset has no i.i.d. reservoir stratum.

Why a tool, not a Final-Answer parser?
--------------------------------------
The agent often reaches multiple distinct findings in one question
("good-class fraud rate is X" + "purpose-code 41 is over-represented").
Recording them mid-loop, one at a time, is more reliable than asking the
model to emit a perfectly-shaped JSON list at the end.  It also produces
the audit trail incrementally -- if the agent crashes after finding #2,
findings #1 and #2 are already on disk.

Tool spec format
----------------
``FINDINGS_TOOL_SPECS`` mirrors ``DICTIONARY_TOOL_SPECS`` and is suitable
for OpenAI native function-calling endpoints.

>>> from ledgerhound.findings import FindingsLog, make_findings_tools
>>> log = FindingsLog("artifacts/findings.jsonl")
>>> tools = make_findings_tools(log, dataset_name="upi_q1_2026")
>>> tools["record_finding"](
...     claim="Fraud rate is 3.2x higher in <30-day-old accounts",
...     sample_frame="within_stratum",
...     evidence_summary="Bucketed by account_age_days",
...     evidence_query="df.groupby('age_bucket')['is_fraud'].mean()",
...     evidence_n_rows_examined=1_000_000,
...     confidence=0.85,
...     priors_tested=["new_accounts_fraud_more"],
...     priors_result="survives",
... )
{'ok': True, 'id': 'f_20260427_a1b2c3d4', 'n_total_findings': 1, 'warnings': []}
"""

from __future__ import annotations

from typing import Any, Callable, Optional, TYPE_CHECKING

from .log import FindingsLog
from .models import Evidence, Finding, PriorResult, SampleFrame

if TYPE_CHECKING:
    from ledgerhound.memory.models import DatasetMemory


# -- Helpers ---------------------------------------------------------------


_VALID_SAMPLE_FRAMES = [e.value for e in SampleFrame]
_VALID_PRIOR_RESULTS = [e.value for e in PriorResult]


def _has_iid_reservoir(dataset_memory: "DatasetMemory | None") -> bool:
    """Return True if the dataset has at least one i.i.d. random stratum.

    "i.i.d. reservoir" = a stratum sampled with simple random sampling and
    no analyst-imposed filters.  This is the only stratum where rates can
    be honestly extrapolated to the population level.
    """
    if dataset_memory is None:
        return False
    sd = getattr(dataset_memory, "sampling_design", None)
    if sd is None or not sd.strata:
        return False
    # Lazy import to avoid circular dependency on memory module.
    from ledgerhound.memory.models import SamplingMethod
    for stratum in sd.strata.values():
        method = stratum.method
        is_iid = (method == SamplingMethod.SIMPLE_RANDOM
                  and not stratum.filters)
        if is_iid:
            return True
    return False


# -- Factory ---------------------------------------------------------------


def make_findings_tools(
    log: FindingsLog,
    dataset_name: Optional[str] = None,
    dataset_memory: "DatasetMemory | None" = None,
) -> dict[str, Callable[..., dict]]:
    """Bind findings tool callables to a log + optional context.

    Parameters
    ----------
    log:
        Destination ``FindingsLog`` (append-only JSONL).
    dataset_name:
        Default ``dataset`` field stamped on every finding when the LLM
        omits it.  Use the same identifier as DatasetMemory.dataset_name
        so the audit trail joins cleanly.
    dataset_memory:
        Optional ``DatasetMemory``.  Powers sample-frame discipline checks:
        a finding tagged ``population_estimate`` against a sample with no
        i.i.d. reservoir gets a non-blocking warning in the response.
    """

    # -- record_finding ----------------------------------------------------

    def record_finding(
        claim: str,
        sample_frame: str,
        evidence_summary: Optional[str] = None,
        evidence_query: Optional[str] = None,
        evidence_code: Optional[str] = None,
        evidence_chart_path: Optional[str] = None,
        evidence_n_rows_examined: Optional[int] = None,
        evidence_result: Optional[Any] = None,
        confidence: float = 0.5,
        sample_frame_note: Optional[str] = None,
        priors_tested: Optional[list[str]] = None,
        priors_result: Optional[str] = None,
        derived_from: Optional[list[str]] = None,
        enables_sampling_for: Optional[list[str]] = None,
        stratum: Optional[str] = None,
        tags: Optional[list[str]] = None,
        dataset: Optional[str] = None,
    ) -> dict:
        """Append a structured Finding to the audit log.

        Required:
            claim          -- one-sentence statement (must be non-empty).
            sample_frame   -- within_stratum | cross_stratum |
                             population_estimate | derived_sample |
                             unspecified.

        Optional (drives the non-fraud sampling design):
            stratum        -- when the claim holds WITHIN a specific slice of
                             the data (sample_frame=within_stratum), name that
                             slice, e.g. "young_accounts" or "high_amount".
                             The non-fraud sampling step reads these to build
                             matched controls per stratum, so a fraud signal
                             that lives in a slice MUST name its stratum here
                             or it cannot seed the sampling design.
            enables_sampling_for -- dataset name(s) this finding should seed a
                             sampling spec for (e.g. "nonfraud").

        Evidence is flattened (``evidence_*`` keys) so the LLM emits flat
        JSON; this is far more reliable than nested objects in practice.
        """
        # -- Validate sample_frame -------------------------------------
        try:
            sf = SampleFrame(sample_frame)
        except ValueError:
            return {
                "ok": False,
                "error": (
                    f"Invalid sample_frame {sample_frame!r}. "
                    f"Must be one of: {_VALID_SAMPLE_FRAMES}"
                ),
            }

        # -- Validate priors_result (optional) -------------------------
        pr: PriorResult = PriorResult.UNTESTED
        if priors_result:
            try:
                pr = PriorResult(priors_result)
            except ValueError:
                return {
                    "ok": False,
                    "error": (
                        f"Invalid priors_result {priors_result!r}. "
                        f"Must be one of: {_VALID_PRIOR_RESULTS}"
                    ),
                }

        # -- Build the Finding -----------------------------------------
        evidence = Evidence(
            summary=evidence_summary or "",
            query=evidence_query,
            code=evidence_code,
            chart_path=evidence_chart_path,
            n_rows_examined=evidence_n_rows_examined,
            result=evidence_result,
        )
        try:
            finding = Finding(
                claim=claim,
                evidence=evidence,
                sample_frame=sf,
                sample_frame_note=sample_frame_note or "",
                confidence=float(confidence),
                dataset=dataset or dataset_name,
                priors_tested=list(priors_tested or []),
                priors_result=pr,
                derived_from=list(derived_from or []),
                enables_sampling_for=list(enables_sampling_for or []),
                stratum=stratum or None,
                tags=list(tags or []),
            )
        except ValueError as e:
            return {"ok": False, "error": f"Invalid finding: {e}"}

        # -- Sample-frame discipline checks (P1.E) ---------------------
        warnings: list[str] = []

        # We can only check the reservoir invariant if we have a sampling
        # design to inspect.  Without a DatasetMemory, stay silent -- the
        # caller may know things we don't.
        if sf == SampleFrame.POPULATION_ESTIMATE and dataset_memory is not None:
            sd = getattr(dataset_memory, "sampling_design", None)
            if sd is not None and sd.strata and not _has_iid_reservoir(dataset_memory):
                warnings.append(
                    "Sample-frame check: this finding is tagged "
                    "'population_estimate' but the dataset has no i.i.d. "
                    "reservoir stratum (simple_random with no filters). "
                    "Population claims need either explicit weighting or "
                    "an unbiased reservoir -- consider re-tagging as "
                    "'cross_stratum' or 'within_stratum'."
                )

        if sf == SampleFrame.UNSPECIFIED:
            warnings.append(
                "Sample-frame is 'unspecified' -- this is allowed but "
                "discouraged. Findings without a frame can't be "
                "interpreted under any sampling design."
            )

        if evidence.is_empty():
            warnings.append(
                "No reproducible evidence (no query / code / chart). "
                "The claim is hearsay until you attach an evidence "
                "trail. Re-record with evidence_query or evidence_code."
            )

        if dataset_memory is not None and priors_tested:
            sd = getattr(dataset_memory, "sampling_design", None)
            encoded = set((sd.priors_encoded if sd else []) or [])
            untested = [p for p in priors_tested if p not in encoded]
            if untested:
                warnings.append(
                    f"priors_tested={untested!r} are not in the dataset's "
                    f"priors_encoded={list(encoded)!r}. Either the prior "
                    f"name is wrong, or you're testing a prior that wasn't "
                    f"baked into the sample (which is allowed but worth "
                    f"flagging in the finding's sample_frame_note)."
                )

        # -- Persist ----------------------------------------------------
        log.append(finding)

        return {
            "ok": True,
            "id": finding.id,
            "sample_frame": finding.sample_frame.value,
            "confidence": finding.confidence,
            "n_total_findings": log.count(),
            "warnings": warnings,
        }

    # -- list_findings ----------------------------------------------------

    def list_findings(limit: int = 10, dataset: Optional[str] = None) -> dict:
        """Peek at recently recorded findings (this session or earlier).

        Use to:
        * Reference earlier findings via ``derived_from`` in a new claim.
        * Avoid re-recording an identical claim.
        * Synthesize a Final Answer that summarises the session.
        """
        ds = dataset or dataset_name
        all_ = log.all(dataset=ds)
        recent = all_[-limit:] if limit > 0 else all_
        return {
            "n_total": len(all_),
            "shown": len(recent),
            "dataset_filter": ds,
            "findings": [
                {
                    "id": f.id,
                    "claim": f.claim,
                    "sample_frame": f.sample_frame.value,
                    "confidence": f.confidence,
                    "priors_result": f.priors_result.value,
                    "tags": list(f.tags),
                    "timestamp": f.timestamp,
                }
                for f in recent
            ],
        }

    return {
        "record_finding": record_finding,
        "list_findings":  list_findings,
    }


# -- OpenAI-style tool specs (for chat-completions ``tools=``) -------------


FINDINGS_TOOL_SPECS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "record_finding",
            "description": (
                "Append a single structured analytical finding to the "
                "audit log. Call this whenever you reach a defensible "
                "conclusion -- do not save findings for the Final Answer. "
                "Every finding MUST declare a sample_frame so claims are "
                "interpretable under the dataset's sampling design."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "claim": {
                        "type": "string",
                        "description": (
                            "One-sentence statement of the finding, with "
                            "magnitude (e.g. '3.2x fraud lift in <30-day "
                            "accounts')."
                        ),
                    },
                    "sample_frame": {
                        "type": "string",
                        "enum": _VALID_SAMPLE_FRAMES,
                        "description": (
                            "within_stratum: claim holds inside one stratum. "
                            "cross_stratum: claim mixes strata with weighting. "
                            "population_estimate: extrapolated to UPI population "
                            "(requires i.i.d. reservoir). "
                            "derived_sample: from an importance-sampled draw "
                            "(Phase 2). unspecified: no frame (discouraged)."
                        ),
                    },
                    "evidence_summary":      {"type": "string"},
                    "evidence_query":        {"type": "string",
                                              "description": "SQL run to produce the result."},
                    "evidence_code":         {"type": "string",
                                              "description": "Python snippet (pandas/cuDF)."},
                    "evidence_chart_path":   {"type": "string",
                                              "description": "Filesystem path to the chart, if any."},
                    "evidence_n_rows_examined": {"type": "integer"},
                    "evidence_result":       {"type": "array",
                                              "items": {"type": "object"},
                                              "description": "Small serialisable result table."},
                    "confidence":            {"type": "number",
                                              "minimum": 0, "maximum": 1,
                                              "default": 0.5},
                    "sample_frame_note":     {"type": "string"},
                    "priors_tested":         {"type": "array",
                                              "items": {"type": "string"}},
                    "priors_result":         {"type": "string",
                                              "enum": _VALID_PRIOR_RESULTS},
                    "derived_from":          {"type": "array",
                                              "items": {"type": "string"},
                                              "description": "IDs of earlier findings this one builds on."},
                    "enables_sampling_for":  {"type": "array",
                                              "items": {"type": "string"},
                                              "description": (
                                                  "Phase-2 hook: datasets whose "
                                                  "future sampling could be derived "
                                                  "from this finding."
                                              )},
                    "tags":                  {"type": "array",
                                              "items": {"type": "string"}},
                    "dataset":               {"type": "string",
                                              "description": (
                                                  "Override default dataset "
                                                  "stamp (rare)."
                                              )},
                },
                "required": ["claim", "sample_frame"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_findings",
            "description": (
                "List recently recorded findings for this dataset.  Use "
                "to reference earlier findings via derived_from, or to "
                "avoid re-recording an identical claim."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit":   {"type": "integer", "default": 10},
                    "dataset": {"type": "string"},
                },
            },
        },
    },
]
