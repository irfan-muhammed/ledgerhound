"""
ledgerhound.memory.bootstrap
=========================

Dataset Memory bootstrap helpers (P1.D).

Three entry points, all returning a ready-to-save :class:`DatasetMemory`:

* :func:`interview` -- Interactive HITL via ``rich.prompt``.  Asks the
  analyst about role, sampling strata, encoded priors, etc.
* :func:`bootstrap_from_yaml` -- Non-interactive draft import.  Lets a
  team check a draft memory YAML into version control and apply it
  with one command.
* :func:`bootstrap_from_data` -- Lightweight inference from a parquet
  or CSV file (schema columns only; everything else is empty).

The interactive path is the primary onboarding experience.  The other
two exist so CI / scripts / tests can exercise the same code path
without a TTY.

Why is this its own module (not in the CLI)?
--------------------------------------------
The interview logic is the actual product feature -- the CLI is just a
thin wrapper.  Keeping the logic library-shaped means we can:

* Unit-test the interview by injecting a fake ``input_fn``.
* Call ``bootstrap_from_data`` from notebooks / pipelines.
* Reuse the same flow when ``ledgerhound eda`` detects a missing memory
  and offers to bootstrap on the fly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from .models import (
    DatasetMemory,
    DatasetRelationship,
    DatasetRole,
    Rollup,
    SamplingDesign,
    SamplingMethod,
    SamplingStratum,
)

log = logging.getLogger(__name__)


# -- Schema inference from data files --------------------------------------


def infer_schema_columns(data_path: str | Path) -> list[str]:
    """Read column names from a parquet, CSV, or TSV file.

    Lightweight -- doesn't load the data, just the schema header.
    Returns an empty list on any failure (so the caller can keep going).
    """
    p = Path(data_path)
    if not p.exists():
        log.warning("Data file not found: %s", p)
        return []

    suffix = p.suffix.lower()
    try:
        if suffix == ".parquet":
            try:
                import pyarrow.parquet as pq
                return list(pq.read_schema(str(p)).names)
            except ImportError:
                # Fall back to pandas if pyarrow isn't directly importable
                import pandas as pd
                return list(pd.read_parquet(p).columns)
        if suffix in (".csv", ".tsv"):
            import pandas as pd
            sep = "\t" if suffix == ".tsv" else ","
            df = pd.read_csv(p, sep=sep, nrows=0)
            return list(df.columns)
        if suffix in (".xlsx", ".xls"):
            import pandas as pd
            df = pd.read_excel(p, nrows=0)
            return list(df.columns)
    except Exception as e:
        log.warning("Schema inference failed for %s: %s", p, e)
    return []


# -- Non-interactive paths -------------------------------------------------


def bootstrap_from_yaml(
    yaml_path: str | Path,
    *,
    overrides: Optional[dict] = None,
) -> DatasetMemory:
    """Load a draft memory YAML and return a DatasetMemory.

    The YAML must contain at least ``dataset_name``.  Anything else is
    optional and gets defaulted.  ``overrides`` is merged on top before
    parsing -- useful for setting a CLI-supplied dataset_name without
    editing the YAML.
    """
    p = Path(yaml_path)
    if not p.exists():
        raise FileNotFoundError(f"Draft YAML not found: {p}")
    with open(p, encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    if overrides:
        d = {**d, **overrides}
    if "dataset_name" not in d or not d["dataset_name"]:
        raise ValueError(f"YAML at {p} is missing required 'dataset_name'")
    return DatasetMemory.from_dict(d)


def bootstrap_from_data(
    dataset_name: str,
    data_path: str | Path,
    *,
    role: DatasetRole = DatasetRole.UNKNOWN,
    description: str = "",
) -> DatasetMemory:
    """Build a minimal DatasetMemory by inferring schema_columns from a file.

    Sampling design, relationships, and rollups are left empty -- the
    analyst can extend later with ``ledgerhound memory init --interactive``
    or by editing the YAML.
    """
    columns = infer_schema_columns(data_path)
    return DatasetMemory(
        dataset_name=dataset_name,
        role=role,
        description=description,
        schema_columns=columns,
        findings_log_path=f".ledgerhound/findings/{dataset_name}.jsonl",
    )


# -- Interactive path ------------------------------------------------------


# Type alias for the input function (overridable for testing)
InputFn = Callable[[str, Optional[str]], str]
ConfirmFn = Callable[[str, bool], bool]
PrintFn = Callable[[str], None]


def _default_input(prompt: str, default: Optional[str] = None) -> str:
    """Default text-input function (uses rich.prompt when available)."""
    try:
        from rich.prompt import Prompt
        return Prompt.ask(prompt, default=default if default is not None else "")
    except ImportError:
        text = input(f"{prompt}{' [' + default + ']' if default else ''}: ")
        return text.strip() or (default or "")


def _default_confirm(prompt: str, default: bool = True) -> bool:
    try:
        from rich.prompt import Confirm
        return Confirm.ask(prompt, default=default)
    except ImportError:
        ans = input(f"{prompt} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
        if not ans:
            return default
        return ans.startswith("y")


def _default_print(text: str) -> None:
    try:
        from rich.console import Console
        Console().print(text)
    except ImportError:
        print(text)


# Valid choices for enum prompts (kept here so the interview module is
# self-contained -- easier to test).
_ROLE_CHOICES = [
    (DatasetRole.FACT,       "Transactions / events (most common)"),
    (DatasetRole.DIMENSION,  "Master / lookup table (one row per entity)"),
    (DatasetRole.ROLLUP,     "Pre-aggregated summary table"),
    (DatasetRole.UNKNOWN,    "Skip / not sure"),
]
_METHOD_CHOICES = [m for m in SamplingMethod]


def _ask_role(input_fn: InputFn, print_fn: PrintFn) -> DatasetRole:
    print_fn("\n[bold]Step 1 -- Dataset role[/bold]")
    for i, (role, descr) in enumerate(_ROLE_CHOICES, 1):
        print_fn(f"  [{i}] {role.value} -- {descr}")
    while True:
        ans = input_fn("Choose role", "1").strip()
        try:
            idx = int(ans) - 1
            if 0 <= idx < len(_ROLE_CHOICES):
                return _ROLE_CHOICES[idx][0]
        except ValueError:
            # Allow typing the value directly
            try:
                return DatasetRole(ans)
            except ValueError:
                pass
        print_fn("[yellow]Invalid choice, try again.[/yellow]")


def _ask_method(input_fn: InputFn, print_fn: PrintFn,
                default: SamplingMethod = SamplingMethod.SIMPLE_RANDOM,
                ) -> SamplingMethod:
    methods_str = " / ".join(m.value for m in _METHOD_CHOICES)
    while True:
        ans = input_fn(f"  Method [{methods_str}]", default.value).strip().lower()
        if not ans:
            return default
        try:
            return SamplingMethod(ans)
        except ValueError:
            print_fn(f"[yellow]Invalid method. Pick one of: {methods_str}[/yellow]")


def _ask_int_or_none(input_fn: InputFn, prompt: str) -> Optional[int]:
    raw = input_fn(prompt, "").strip()
    if not raw:
        return None
    try:
        return int(raw.replace(",", "").replace("_", ""))
    except ValueError:
        return None


def _ask_float_or_none(input_fn: InputFn, prompt: str) -> Optional[float]:
    raw = input_fn(prompt, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _ask_list(input_fn: InputFn, print_fn: PrintFn, item_label: str,
              ) -> list[str]:
    """Prompt repeatedly until the analyst submits an empty line."""
    items: list[str] = []
    print_fn(f"  Enter one {item_label} per line; empty line to finish.")
    while True:
        item = input_fn(f"    > {item_label}", "").strip()
        if not item:
            break
        items.append(item)
    return items


def _interview_strata(input_fn, confirm_fn, print_fn) -> dict:
    """Collect strata one at a time."""
    strata: dict[str, SamplingStratum] = {}
    print_fn("\n[bold]Step 3a -- Sampling strata[/bold] "
             "(name each stratum; empty name to finish)")
    while True:
        name = input_fn("Stratum name (empty to finish)", "").strip()
        if not name:
            break
        if name in strata:
            print_fn(f"[yellow]Stratum {name!r} already defined; "
                     f"editing existing.[/yellow]")
        method = _ask_method(input_fn, print_fn)
        n_sample = _ask_int_or_none(input_fn, "  n_sample")
        n_population = _ask_int_or_none(input_fn, "  n_population")
        p_inclusion = _ask_float_or_none(input_fn, "  p_inclusion")
        filters = _ask_list(input_fn, print_fn, "filter")
        strata[name] = SamplingStratum(
            name=name, method=method,
            n_sample=n_sample, n_population=n_population,
            p_inclusion=p_inclusion, filters=filters,
        )
    return strata


def interview(
    dataset_name: str,
    *,
    data_path: Optional[str | Path] = None,
    input_fn: Optional[InputFn] = None,
    confirm_fn: Optional[ConfirmFn] = None,
    print_fn: Optional[PrintFn] = None,
) -> DatasetMemory:
    """Run the interactive bootstrap interview and return a DatasetMemory.

    Parameters
    ----------
    dataset_name:
        Identifier for this dataset (also the YAML filename stem).
    data_path:
        Optional path to a parquet / CSV file.  When provided, schema_columns
        is inferred automatically -- the analyst doesn't have to type them.
    input_fn / confirm_fn / print_fn:
        Override I/O for testing.  Defaults use ``rich.prompt`` /
        ``rich.console`` so the CLI experience is nice; tests substitute
        scripted versions.
    """
    input_fn = input_fn or _default_input
    confirm_fn = confirm_fn or _default_confirm
    print_fn = print_fn or _default_print

    print_fn(f"\n[bold cyan]Ledgerhound Memory bootstrap[/bold cyan] -- "
             f"dataset: [bold]{dataset_name}[/bold]\n")

    # -- Step 1: Role ------------------------------------------------------
    role = _ask_role(input_fn, print_fn)

    # -- Step 2: Description -----------------------------------------------
    print_fn("\n[bold]Step 2 -- Description[/bold] (one line, optional)")
    description = input_fn("Description", "").strip()

    # -- Step 3: Sampling design ------------------------------------------
    print_fn("\n[bold]Step 3 -- Sampling design[/bold]")
    sampling_design: Optional[SamplingDesign] = None
    if confirm_fn("Is this a sampled dataset (vs. full population)?", True):
        strata = _interview_strata(input_fn, confirm_fn, print_fn)
        print_fn("\n[bold]Step 3b -- Encoded priors[/bold] "
                 "(analyst assumptions baked into the sample, e.g. "
                 "'new_accounts_fraud_more')")
        priors_encoded = _ask_list(input_fn, print_fn, "prior")
        print_fn("\n[bold]Step 3c -- Stale-if conditions[/bold] "
                 "(when this design becomes invalid; optional)")
        stale_if = _ask_list(input_fn, print_fn, "stale_if condition")
        sampling_design = SamplingDesign(
            strata=strata, priors_encoded=priors_encoded, stale_if=stale_if,
        )

    # -- Step 4: Schema columns -------------------------------------------
    print_fn("\n[bold]Step 4 -- Schema columns[/bold]")
    schema_columns: list[str] = []
    if data_path:
        schema_columns = infer_schema_columns(data_path)
        if schema_columns:
            print_fn(f"  Inferred {len(schema_columns)} columns from "
                     f"[cyan]{data_path}[/cyan]")
        else:
            print_fn(f"  [yellow]Could not infer columns from {data_path}.[/yellow]")
    if not schema_columns:
        if confirm_fn("Add schema columns by hand?", False):
            schema_columns = _ask_list(input_fn, print_fn, "column name")

    # -- Step 5: Findings log path ----------------------------------------
    default_findings = f".ledgerhound/findings/{dataset_name}.jsonl"
    print_fn("\n[bold]Step 5 -- Findings log path[/bold]")
    findings_log_path = input_fn(
        "Findings log path", default_findings,
    ).strip() or default_findings

    return DatasetMemory(
        dataset_name=dataset_name,
        role=role,
        description=description,
        schema_columns=schema_columns,
        sampling_design=sampling_design,
        findings_log_path=findings_log_path,
    )


# -- Listing / discovery ---------------------------------------------------


def list_datasets(base_dir: str | Path = ".ledgerhound/memory") -> list[str]:
    """Return dataset_names of every memory YAML found under ``base_dir``."""
    p = Path(base_dir)
    if not p.exists():
        return []
    return sorted(f.stem for f in p.glob("*.yaml") if f.is_file())
