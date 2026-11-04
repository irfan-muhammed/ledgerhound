"""
ledgerhound.profiler.detectors.column_classifier
==============================================

Classifies each column by semantic type, role, and PII status.
Uses schema metadata when provided, falls back to heuristics.

Phase B fixes
-------------
* **Mixed hash detection**: praccno has both 32-char MD5 and 64-char
  SHA256.  Now allows up to 2 distinct hash lengths.
* **Partial hash detection**: payee_verified_name is hashed for P2P
  but plain text for P2M.  Flagged as HASH with ``mixed_content`` tag.
* **Excel serial date**: asdt = 45762 (Excel serial).  Detected by
  name hint + 5-digit integer range 38000-55000.
* **Device fingerprint**: prdid contains non-hex chars (UK0, AZY).
  Classified as ENTITY_ID, not mistaken for hash.
"""

from __future__ import annotations

import re
from typing import Any

from ledgerhound.backends.base import ColumnInfo
from ledgerhound.schema.models import (
    ColumnRole, ColumnSchema, PIIStatus, SemanticType,
)

_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")
_HASH_LENGTHS = {32, 40, 64, 128}  # MD5, SHA1, SHA256, SHA512


class ColumnClassifier:
    """Merge user-provided schema with runtime heuristics."""

    def classify(self, col_info: ColumnInfo, samples: list[Any],
                 schema_col: ColumnSchema | None = None) -> ColumnSchema:
        """Return a fully-populated ColumnSchema for one column."""

        if schema_col and schema_col.provided:
            result = ColumnSchema(
                name=col_info.name,
                dtype=col_info.dtype if col_info.dtype != "object" else schema_col.dtype,
                description=schema_col.description,
                sample_values=samples, provided=True,
                links_to=schema_col.links_to,
                transform=schema_col.transform,
                display_unit=schema_col.display_unit,
                tags=list(schema_col.tags),
            )
            result.semantic_type = (
                schema_col.semantic_type
                if schema_col.semantic_type != SemanticType.UNKNOWN
                else _infer_semantic(col_info, samples))
            result.pii_status = (
                schema_col.pii_status
                if schema_col.pii_status != PIIStatus.NONE
                else _detect_pii(col_info.name, samples, result.semantic_type))
            result.role = (
                schema_col.role
                if schema_col.role != ColumnRole.UNKNOWN
                else _infer_role(col_info.name, schema_col.description, result.semantic_type))
            if result.semantic_type == SemanticType.HASH and _is_mixed_content(samples):
                if "mixed_content" not in result.tags:
                    result.tags.append("mixed_content")
            return result

        # Full inference
        sem = _infer_semantic(col_info, samples)
        tags = []
        if sem == SemanticType.HASH and _is_mixed_content(samples):
            tags.append("mixed_content")

        result = ColumnSchema(
            name=col_info.name, dtype=col_info.dtype,
            semantic_type=sem,
            role=_infer_role(col_info.name, "", sem),
            pii_status=_detect_pii(col_info.name, samples, sem),
            sample_values=samples, provided=False, tags=tags,
        )
        if _is_excel_serial_date(col_info.name, samples):
            result.semantic_type = SemanticType.TEMPORAL
            result.transform = "excel_serial_date"
            result.role = ColumnRole.TIMESTAMP
        return result


# -- semantic type inference --------------------------------

def _infer_semantic(ci: ColumnInfo, samples: list[Any]) -> SemanticType:
    dtype, name = str(ci.dtype).lower(), ci.name.lower()

    if _is_hash(samples):
        return SemanticType.HASH
    if _is_partial_hash(samples):
        return SemanticType.HASH
    if _is_excel_serial_date(name, samples):
        return SemanticType.TEMPORAL
    if "datetime" in dtype or "timestamp" in dtype:
        return SemanticType.TEMPORAL
    if any(kw in name for kw in ("date", "time", "created", "updated")) and "prdid" not in name:
        return SemanticType.TEMPORAL
    if any(t in dtype for t in ("int", "float", "double", "decimal")):
        return SemanticType.NUMERIC
    if ci.unique <= 2 and ci.non_null > 0:
        return SemanticType.BINARY
    if "object" in dtype or "str" in dtype:
        if _is_device_fingerprint(name, samples):
            return SemanticType.IDENTIFIER
        if ci.non_null > 0 and (ci.unique / max(ci.non_null, 1)) > 0.9:
            return SemanticType.IDENTIFIER
        return SemanticType.CATEGORICAL
    return SemanticType.UNKNOWN


# -- hash detection -----------------------------------------

def _is_hash(samples: list[Any]) -> bool:
    """Pure hex-hash: allows up to 2 distinct lengths (MD5 + SHA256 mix)."""
    valid = [s for s in samples if isinstance(s, str) and s.strip()]
    if len(valid) < 2:
        return False
    lengths = {len(s) for s in valid}
    if len(lengths) > 2 or not lengths <= _HASH_LENGTHS:
        return False
    return all(_HEX_RE.match(s) for s in valid)


def _is_partial_hash(samples: list[Any]) -> bool:
    """Mixed hash + plain text (e.g. payee_verified_name).

    Returns True if 50%-99% of non-empty samples are hex hashes.
    If 100%, _is_hash() already caught it.
    """
    valid = [s for s in samples if isinstance(s, str) and s.strip()]
    if len(valid) < 3:
        return False
    hashes = sum(1 for s in valid if len(s) in _HASH_LENGTHS and _HEX_RE.match(s))
    ratio = hashes / len(valid)
    return 0.5 < ratio < 1.0


def _is_mixed_content(samples: list[Any]) -> bool:
    """Check if a hash column also has non-hash values."""
    valid = [s for s in samples if isinstance(s, str) and s.strip()]
    if len(valid) < 2:
        return False
    hashes = sum(1 for s in valid if len(s.strip()) in _HASH_LENGTHS and _HEX_RE.match(s.strip()))
    return 0 < hashes < len(valid)


# -- special format detection -------------------------------

def _is_excel_serial_date(name: str, samples: list[Any]) -> bool:
    """Excel serial date: column named *date*/*dt* with 5-digit ints ~40000-55000."""
    if not any(kw in name.lower() for kw in ("date", "dt", "asdt")):
        return False
    nums = []
    for s in samples:
        try:
            nums.append(int(float(str(s))))
        except (ValueError, TypeError):
            pass
    return len(nums) >= 2 and all(38000 <= n <= 55000 for n in nums)


def _is_device_fingerprint(name: str, samples: list[Any]) -> bool:
    """Device IDs: high-cardinality alphanum with non-hex chars."""
    if not any(kw in name.lower() for kw in ("prdid", "deviceid", "device_id")):
        return False
    valid = [s for s in samples if isinstance(s, str) and len(s) > 10]
    if len(valid) < 2:
        return False
    return any(not _HEX_RE.match(s) and any(c.isalpha() for c in s) for s in valid)


# -- PII and role inference ---------------------------------

def _detect_pii(name: str, samples: list, sem: SemanticType) -> PIIStatus:
    if sem == SemanticType.HASH:
        return PIIStatus.HASHED
    pii_kw = ("accno", "accname", "mobile", "phone", "email", "name", "pan", "aadhaar")
    if any(kw in name.lower() for kw in pii_kw):
        return PIIStatus.RAW
    return PIIStatus.NONE


def _infer_role(name: str, desc: str, sem: SemanticType) -> ColumnRole:
    n, d = name.lower(), desc.lower()
    if n in ("txnid", "txn_id", "id") or "transaction id" in d:
        return ColumnRole.PRIMARY_KEY
    if "amount" in n or "amount" in d:
        return ColumnRole.AMOUNT
    if sem == SemanticType.TEMPORAL:
        return ColumnRole.TIMESTAMP
    if n.endswith("respcode") or "response code" in d:
        return ColumnRole.RESPONSE_CODE
    if "status" in n:
        return ColumnRole.STATUS
    if any(kw in n for kw in ("fraud", "target", "label", "is_fraud")):
        return ColumnRole.TARGET
    if any(kw in n for kw in ("vpa", "vaddr", "mobile", "prdid", "deviceid")):
        return ColumnRole.ENTITY_ID
    return ColumnRole.FEATURE
