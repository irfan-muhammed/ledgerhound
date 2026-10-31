"""
sentinel.dictionary.loader
==========================

Parsers for the three org-supplied source files plus auto-alignment
between dictionary names and DB names.

Inputs (defaulted to ``sentinel/dictionary/data/``):

1. ``upi_ft_dictionary.csv``   -- column registry with M / NM / NA codes
                                   and analyst comments (the *quality* signal).
2. ``upi_columns_all.txt``     -- actual DB column names discovered via DESCRIBE.
3. ``upi_purpose_codes.csv``   -- purpose-code -> description + product table.

Quality severity heuristic
--------------------------
Population status drives quality severity:

==========  =====================================  ====================
Source code Meaning                                Default severity
==========  =====================================  ====================
M           Populated correctly                    INFO (only if comment)
NM          Populated wrongly / partially          HIGH
NA          Not available                          CRITICAL
==========  =====================================  ====================

Honest-engineering note
-----------------------
This loader is **deliberately conservative**: it never invents column
relationships beyond name-similarity heuristics. Anything more -- e.g.
"these two columns join on this key" -- must be confirmed by a human
through the HITL flow before it lands in the KB. We don't want the bias
of an over-confident loader corrupting the truth set the agent reads from.
"""

from __future__ import annotations

import ast
import csv
import logging
import re
from pathlib import Path
from typing import Iterable, Optional

from .models import (
    ColumnInfo,
    DictionaryStatus,
    PopulationStatus,
    QualityFlag,
    QualitySeverity,
    ReferenceCode,
    RelationKind,
    Relationship,
)
from .store import SchemaKB

log = logging.getLogger(__name__)


# Default paths inside the package
_DATA_DIR = Path(__file__).parent / "data"
DEFAULT_DICTIONARY_CSV   = _DATA_DIR / "upi_ft_dictionary.csv"
DEFAULT_COLUMNS_TXT      = _DATA_DIR / "upi_columns_all.txt"
DEFAULT_PURPOSE_CODES_CSV = _DATA_DIR / "upi_purpose_codes.csv"


# -- Semantic-type inference (simple, transparent rules) --


_SEMANTIC_RULES: list[tuple[re.Pattern, str]] = [
    # Order matters: more-specific patterns first.
    (re.compile(r"(rrn|txn(_)?id|txnkey|txn_key|umn)$",          re.I), "transaction_id"),
    (re.compile(r"(_id|_seqnum|infoid|sid)$",                     re.I), "identifier"),
    (re.compile(r"(account_number|accno|aadhar|mobilenumber)$",  re.I), "pii_identifier"),
    (re.compile(r"(_amount|_amt|amount(_final)?|baseamount)$",    re.I), "amount"),
    (re.compile(r"(timestamp|_ts|asdt|trndate|_date|invoicedate)$", re.I), "temporal"),
    (re.compile(r"(geocode|_location|_address)$",                 re.I), "geo"),
    (re.compile(r"(ifsc|mcc(_code)?|bin|mmid|psp|app_id|handle)$", re.I), "code"),
    (re.compile(r"(_name|_brand|_genre|originatorname)$",         re.I), "text_identity"),
    (re.compile(r"(_status|success_flag|_flag|_type|_subtype|cred_type|cred_subtype)$", re.I), "categorical"),
    (re.compile(r"(respcd|respcode|response|error_code|finalrespcode|reversal_respcode_)", re.I), "response_code"),
    (re.compile(r"(riskscore|score)$",                            re.I), "score"),
    (re.compile(r"(device_id|os_type|client|port|windowid|imps|telecom)$", re.I), "device_or_channel"),
    (re.compile(r"(initiation_mode|initiating_channel|stages|purposecode|purpose|refcategory)$", re.I), "categorical"),
    (re.compile(r"(read_timeout|mis_crtn|apiversion|siteid|umn|fx|mkup|orgtxnid)$", re.I), "metadata"),
]


def _infer_semantic_type(name: str) -> Optional[str]:
    """Best-effort semantic type from name pattern alone.

    Returns None if no pattern matches -- caller may then leave it blank
    or set it via a later HITL step.
    """
    n = name.lower()
    for pat, sem in _SEMANTIC_RULES:
        if pat.search(n):
            return sem
    return None


def _infer_role(name: str, semantic_type: Optional[str]) -> Optional[str]:
    """Best-effort fraud-modeling role from name + semantic type."""
    n = name.lower()
    if semantic_type == "amount":
        return "amount"
    if semantic_type == "temporal":
        return "timestamp"
    if semantic_type in ("transaction_id", "identifier"):
        return "entity_id"
    if semantic_type == "pii_identifier":
        return "entity_id"
    if "fraud" in n or "success_flag" in n:
        return "target"
    if semantic_type == "response_code":
        return "response_code"
    if semantic_type in ("categorical", "code", "geo", "score", "device_or_channel"):
        return "feature"
    return None


# -- Status / population mapping --


_STATUS_MAP = {
    "mandatory": DictionaryStatus.MANDATORY,
    "optional":  DictionaryStatus.OPTIONAL,
    "":          DictionaryStatus.NEW,         # blank STATUS column -> New_Field
    "new":       DictionaryStatus.NEW,
}

_POPULATION_MAP = {
    "M":  PopulationStatus.POPULATED,
    "NM": PopulationStatus.NOT_POPULATED,
    "NA": PopulationStatus.NOT_AVAILABLE,
    "":   PopulationStatus.UNKNOWN,
}


def _severity_for(population: PopulationStatus, has_comment: bool) -> QualitySeverity:
    """Pick severity for a quality flag based on population status."""
    if population == PopulationStatus.NOT_AVAILABLE:
        return QualitySeverity.CRITICAL
    if population == PopulationStatus.NOT_POPULATED:
        return QualitySeverity.HIGH
    if has_comment:
        return QualitySeverity.INFO
    return QualitySeverity.LOW


# -- Loaders ----------------------------------------------


def load_dictionary_csv(
    kb: SchemaKB,
    path: str | Path = DEFAULT_DICTIONARY_CSV,
    dataset: str = "upi_ft",
) -> int:
    """Load the org's ``upi_ft_Data_Dictionary_V0.csv`` into the KB.

    Each row becomes a ColumnInfo; non-trivial comments on NM/NA columns
    become QualityFlags. Returns the number of columns ingested.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dictionary CSV not found: {path}")

    n_cols = 0
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # noqa: F841 -- header is fixed shape
        for row in reader:
            # Skip blank trailing lines
            if not any(cell.strip() for cell in row):
                continue
            # Pad short rows
            row = (row + ["", "", "", ""])[:4]
            status_raw, field_name, pop_raw, comment = (c.strip() for c in row)

            if not field_name:
                continue

            status = _STATUS_MAP.get(status_raw.lower(), DictionaryStatus.UNKNOWN)
            # The data dictionary itself contains a quirky entry:
            # "error_code - Not Available" -- we strip the suffix.
            field_name = field_name.split(" - ")[0].strip()

            population = _POPULATION_MAP.get(pop_raw.upper(), PopulationStatus.UNKNOWN)
            semantic = _infer_semantic_type(field_name)
            role = _infer_role(field_name, semantic)

            col = ColumnInfo(
                canonical_name=field_name,
                dict_name=field_name,
                dataset=dataset,
                semantic_type=semantic,
                role=role,
                status=status,
                population=population,
                description="",          # the source dict has no description column
                comments=comment,
                is_active=(population != PopulationStatus.NOT_AVAILABLE),
            )
            kb.add_column(col)
            n_cols += 1

            # Promote meaningful comments to quality flags
            if comment:
                kb.add_quality_flag(QualityFlag(
                    canonical_name=field_name,
                    issue=comment,
                    severity=_severity_for(population, has_comment=True),
                    source_quote=comment,
                    source_file=path.name,
                ))

    log.info("Loaded %d columns from %s", n_cols, path.name)
    return n_cols


def load_columns_txt(
    kb: SchemaKB,
    path: str | Path = DEFAULT_COLUMNS_TXT,
    dataset: str = "upi_ft",
) -> int:
    """Load the actual DB column names from ``upi_columns_all.txt``.

    The file is a Python module fragment containing
    ``COLUMNS_ALL = [...]``. We extract the list with ``ast.literal_eval``
    and ensure every name has a column entry.

    Naming-drift handling
    ---------------------
    Critical: dictionary loads name like ``txn_id``; this loader sees
    ``txnid`` from the DB. We must NOT create two canonical rows for what
    is one column. We resolve via a *normalised-name index* (lowercase,
    no underscores) built once from existing columns. Match -> register as
    an alias on the existing canonical row, no new row created.

    Returns the number of names processed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Columns file not found: {path}")

    text = path.read_text(encoding="utf-8")
    m = re.search(r"COLUMNS_ALL\s*=\s*(\[[\s\S]*?\])", text)
    if not m:
        raise ValueError(f"Could not find COLUMNS_ALL list in {path}")
    columns: list[str] = ast.literal_eval(m.group(1))

    # Build a normalised-name -> canonical_name index of EXISTING columns.
    # We seed it with whatever the dictionary load already produced.
    norm_index: dict[str, str] = {}
    for c in kb.all_columns(dataset=dataset, active_only=False):
        for cand in {c.canonical_name, c.db_alias, c.dict_name}:
            if cand:
                norm_index.setdefault(_normalise(cand), c.canonical_name)

    n = 0
    for db_name in columns:
        norm = _normalise(db_name)
        existing_canonical = norm_index.get(norm)

        if existing_canonical is not None:
            # We already know this column under another name -- link, don't duplicate.
            existing = kb.lookup(existing_canonical)
            if existing is not None and not existing.db_alias:
                existing.db_alias = db_name
                kb.add_column(existing)   # upsert; preserves dict_name etc.
            kb.add_alias(existing_canonical, db_name, source="db_columns")
        else:
            # Pure DB-only column (no dictionary entry) -- fresh record.
            semantic = _infer_semantic_type(db_name)
            role = _infer_role(db_name, semantic)
            kb.add_column(ColumnInfo(
                canonical_name=db_name,
                db_alias=db_name,
                dataset=dataset,
                semantic_type=semantic,
                role=role,
                status=DictionaryStatus.UNKNOWN,
                population=PopulationStatus.UNKNOWN,
                description="",
                comments="(present in DB; not in source dictionary)",
                is_active=True,
            ))
            norm_index[norm] = db_name
        n += 1

    log.info("Processed %d DB column names from %s", n, path.name)
    return n


def load_purpose_codes_csv(
    kb: SchemaKB,
    path: str | Path = DEFAULT_PURPOSE_CODES_CSV,
) -> int:
    """Load purpose-code lookup table.

    The source CSV has quirks:
      * Range entries like ``32-40`` (reserved blocks) -- stored verbatim.
      * Trailing empty fields from spreadsheet export -- ignored.
      * Occasional duplicate codes (e.g. two ``52`` entries) -- last write wins,
        with a warning logged for auditability.

    Returns the number of entries ingested.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Purpose codes CSV not found: {path}")

    n = 0
    seen: set[tuple[str, str]] = set()
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if not any(c.strip() for c in row):
                continue
            row = (row + ["", "", ""])[:3]
            code, description, product = (c.strip() for c in row)
            if not code:
                continue

            # Some rows have a description in the "product" column when the
            # description column is blank (e.g. "Reserved for ..." rows).
            if not description and product:
                description, product = product, ""

            key = (code, "purpose_code")
            if key in seen:
                log.warning("Duplicate purpose_code %r -- overwriting", code)
            seen.add(key)

            kb.add_reference_code(ReferenceCode(
                code=code,
                code_type="purpose_code",
                description=description,
                product=product or None,
            ))
            n += 1

    log.info("Loaded %d purpose codes from %s", n, path.name)
    return n


# -- Auto-align: dictionary <-> DB column names --


def _normalise(name: str) -> str:
    """Strip underscores and lowercase -- the laxest reasonable match."""
    return re.sub(r"[_\s\-]+", "", name).lower()


def auto_align_aliases(kb: SchemaKB, dataset: str = "upi_ft") -> int:
    """Heuristically link dictionary names to DB names by string similarity.

    Conservative: only registers an alias when the *normalised* form
    (lowercase, no underscores) matches **exactly**. Anything fuzzier than
    this should go through HITL -- we won't guess.

    Examples that link automatically:
        ``txn_id``                 <-> ``txnid``
        ``payer_account_number``   <-> ``payer_account_number`` (already same)
        ``txn_subtype``            <-> ``txnsubtype``

    Examples that *don't* link (intentionally):
        ``payer_account_number``   <-> ``praccno_nz``  (semantically related,
                                                       but text doesn't match)
        ``txn_status``             <-> ``finalrespcode`` (similar role, different
                                                          name space)

    Returns the number of new aliases added.
    """
    all_cols = kb.all_columns(dataset=dataset, active_only=False)
    by_norm: dict[str, list[ColumnInfo]] = {}
    for c in all_cols:
        for cand in {c.canonical_name, c.db_alias, c.dict_name}:
            if cand:
                by_norm.setdefault(_normalise(cand), []).append(c)

    n_added = 0
    for norm, cols in by_norm.items():
        if len(cols) <= 1:
            continue
        # Multiple ColumnInfo records share the same normalised form.
        # Pick the dictionary-defined record as canonical (preferred), else
        # the one with non-blank status, else the first.
        cols_sorted = sorted(
            cols,
            key=lambda x: (
                x.dict_name is None,
                x.status == DictionaryStatus.UNKNOWN,
                x.canonical_name,
            ),
        )
        canonical = cols_sorted[0]
        for other in cols_sorted[1:]:
            if other.canonical_name == canonical.canonical_name:
                continue
            kb.add_alias(canonical.canonical_name, other.canonical_name, source="auto_normalise")
            kb.add_relationship(Relationship(
                col_a=canonical.canonical_name,
                col_b=other.canonical_name,
                kind=RelationKind.SAME_ENTITY,
                notes="auto-aligned by normalised name",
            ))
            n_added += 1

    log.info("Auto-aligned %d alias pairs in dataset %r", n_added, dataset)
    return n_added


# -- Convenience: load all defaults in one call --


def load_defaults(
    kb: SchemaKB,
    dataset: str = "upi_ft",
    dictionary_csv: str | Path = DEFAULT_DICTIONARY_CSV,
    columns_txt: str | Path = DEFAULT_COLUMNS_TXT,
    purpose_codes_csv: str | Path = DEFAULT_PURPOSE_CODES_CSV,
    align: bool = True,
) -> dict:
    """One-shot load: data dictionary + DB column names + purpose codes.

    Returns a small dict with counts for ``sentinel dictionary load`` to
    print back to the user.
    """
    n_dict = load_dictionary_csv(kb, dictionary_csv, dataset=dataset)
    n_cols = load_columns_txt(kb, columns_txt, dataset=dataset)
    n_codes = load_purpose_codes_csv(kb, purpose_codes_csv)
    n_aligned = auto_align_aliases(kb, dataset=dataset) if align else 0
    return {
        "dictionary_columns": n_dict,
        "db_columns": n_cols,
        "purpose_codes": n_codes,
        "auto_aligned": n_aligned,
        **kb.stats(),
    }
