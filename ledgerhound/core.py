"""
ledgerhound.core
=============

Main entry point -- wires backends, schema, profiler, knowledge, agent.
Now with adaptive compute routing (GPU/CPU) and bank_fraud preset.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from ledgerhound.backends.pandas_backend import PandasBackend
from ledgerhound.schema.ingester import SchemaIngester
from ledgerhound.schema.models import DatasetSchema, TableSchema, ColumnSchema
from ledgerhound.profiler.engine import ProfilerEngine
from ledgerhound.profiler.digest import Digest
from ledgerhound.knowledge.loader import KnowledgeBase


def _product_version() -> str:
    try:
        from ledgerhound import __version__
        return __version__
    except Exception:
        return ""


def _suite_version() -> str:
    try:
        from importlib.metadata import version
        return version("llmfe-suite")
    except Exception:
        return ""


def _data_shape_of(data) -> list:
    """Best-effort data shape for the telemetry envelope (never raises)."""
    try:
        if hasattr(data, "shape"):
            return list(data.shape)
        import os as _os
        from pathlib import Path as _Path
        if isinstance(data, (str, _Path)) and _os.path.exists(str(data)):
            import pandas as _pd
            ext = str(data).lower().rsplit(".", 1)[-1]
            if ext == "parquet":
                import pyarrow.parquet as _pq
                md = _pq.ParquetFile(str(data)).metadata
                return [md.num_rows, md.num_columns]
            return list(_pd.read_csv(str(data)).shape)
    except Exception:
        pass
    return []


class Ledgerhound:
    """Top-level API -- profile data and run agentic EDA."""

    _PRESETS = {
        "upi": "upi_schema.json",
        "bank_fraud": "bank_fraud_schema.json",
    }

    def __init__(self, data=None, schema=None, preset=None, rules=None,
                 target=None, model=None, model_name=None, api_key=None,
                 llm_config=None, knowledge_dir=None, source=None):
        self._data_source = data
        self._target = target
        self._compute = None
        # `source` records where an in-memory `data` frame came from. Without
        # it, profile()'s provenance field would be str(DataFrame) -- i.e.
        # the whole table rendered into the digest. Ignored when `data` is
        # already a path (the path IS the provenance).
        self._source = source

        # LLM config captured up front so the compute router knows whether a
        # local LLM will occupy a GPU (and must be reserved one) before it
        # allocates GPUs to the data backend.
        self._llm = None
        self._llm_url = model
        self._llm_name = model_name
        self._llm_key = api_key
        self._llm_cfg_path = llm_config
        self._eda = None

        # Telemetry attribution via Ledgerhound's OWN identity (never prompts,
        # never blocks — current_user() degrades to OS user/anonymous).
        # The work access gate (identity.ensure_login) remains
        # upstream behavior and is untouched by telemetry.
        self._telemetry = None
        try:
            from ledgerhound.telemetry import TelemetryClient, set_global_client
            from ledgerhound.identity import current_user as _current_user
            _user = _current_user()
            self._telemetry = TelemetryClient(
                user=_user, product="ledgerhound",
                suite_version=_suite_version(), product_version=_product_version())
            set_global_client(self._telemetry)
            self._telemetry.fire_event("run_started", {
                "mode": "library",
                "data_shape": _data_shape_of(data),
                "target": target or "",
                "preset": preset or "",
            })
        except Exception:
            self._telemetry = None  # telemetry must never block a run

        # Backend (with compute routing)
        self._backend = None
        if data is not None:
            data_path = Path(str(data)) if isinstance(data, str) else None
            if data_path and not data_path.exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")
            self._backend, self._compute = self._init_backend(data, data_path)

        # Schema
        self._schema = DatasetSchema()
        if schema is not None:
            if isinstance(schema, (str, Path)):
                p = Path(schema)
                if not p.exists():
                    raise FileNotFoundError(
                        f"Schema file not found: {p}\n"
                        f"Hint: use --preset upi or --preset bank_fraud")
            self._schema = SchemaIngester().ingest(schema)
        elif preset is not None:
            self._schema = self._load_preset(preset)

        # Knowledge. This package must never carry real customer/production
        # data or proprietary business rules (it ships in the public repo)
        # -- only generic, public defaults live under knowledge/defaults.
        # An institutional knowledge directory (real fraud rationale,
        # column semantics, business rules) is loaded from
        # main_config.yaml's knowledge.knowledge_dir (or the older
        # LEDGERHOUND_KNOWLEDGE_DIR env var) if set, layered on top -- same
        # override pattern as ledgerhound.context.domain_profiles.
        from ledgerhound.config import get_str
        default_kb = Path(__file__).parent / "knowledge" / "defaults"
        self._knowledge = KnowledgeBase(knowledge_dir or default_kb)
        if not knowledge_dir:
            override = get_str("knowledge", "knowledge_dir", env="LEDGERHOUND_KNOWLEDGE_DIR")
            if override and Path(override).is_dir():
                self._knowledge.load(Path(override))
        if rules:
            p = Path(rules)
            if p.exists():
                self._knowledge.rules.update(
                    yaml.safe_load(p.read_text(encoding="utf-8")) or {})

    def _init_backend(self, data, data_path):
        """Initialize backend with adaptive compute routing."""
        compute = None
        try:
            from ledgerhound.backends.compute_router import ComputeRouter, get_backend
            router = ComputeRouter()
            # A local LLM (HF model or local server) occupies a GPU, so the
            # router must reserve one for it; a remote API (gateway Kimi etc.)
            # leaves every GPU for cuDF. Decide from the config CONTENT
            # (provider/base_url), not the file name.
            llm_is_local, llm_gpu = False, 0
            if self._llm_cfg_path and Path(self._llm_cfg_path).exists():
                try:
                    with open(self._llm_cfg_path, encoding="utf-8") as _f:
                        _cfg = yaml.safe_load(_f) or {}
                    llm_is_local, llm_gpu = router.placement_from_config(_cfg)
                except Exception:
                    pass
            elif self._llm_url:
                llm_is_local = router.infer_llm_is_local(self._llm_url)
            compute = router.detect(data_path, llm_is_local=llm_is_local,
                                    llm_gpu=llm_gpu)
            try:
                from ledgerhound.output.banner import print_compute_decision
                print_compute_decision(compute)
            except Exception:
                from rich.console import Console
                Console().print(f"[dim]Compute: {compute.reason}[/dim]")
            backend = get_backend(data, compute)
            return backend, compute
        except ImportError:
            pass
        except Exception:
            pass
        return PandasBackend(data), compute

    def _source_label(self) -> str:
        """Provenance string for the digest: the explicit `source=` if given,
        else the path -- never str() of an in-memory frame."""
        if self._source:
            return str(self._source)
        if isinstance(self._data_source, (str, Path)):
            return str(self._data_source)
        return "<in-memory DataFrame>"

    def profile(self, target: str | None = None, parallel: bool = False,
                n_workers: int = 3) -> Digest:
        if self._backend is None:
            raise ValueError("No data source. Pass data= to Ledgerhound().")
        if not self._schema.tables:
            self._schema = self._auto_schema()
        try:
            digest = ProfilerEngine(self._backend, self._schema).run(
                target_col=target or self._target,
                source=self._source_label(),
                parallel=parallel, n_workers=n_workers)
        except Exception as e:
            if self._telemetry is not None:
                try:
                    self._telemetry.fire_event("error", {
                        "phase": "profile",
                        "error_type": type(e).__name__,
                        "error_message": str(e)[:500],
                    })
                except Exception:
                    pass
            raise
        if self._telemetry is not None:
            try:
                self._telemetry.fire_event("run_finished", {
                    "phase": "profile",
                    "target": target or self._target or "",
                    "final_status": "ok",
                })
            except Exception:
                pass
        return digest

    def _auto_schema(self) -> DatasetSchema:
        cols = {ci.name: ColumnSchema(name=ci.name, dtype=str(ci.dtype))
                for ci in self._backend.get_columns()}
        t = TableSchema(name="primary", alias="primary", columns=cols)
        return DatasetSchema(tables={"primary": t}, primary_table="primary")

    def _load_preset(self, name: str) -> DatasetSchema:
        import json
        filename = self._PRESETS.get(name.lower())
        if filename is None:
            raise ValueError(f"Unknown preset '{name}'. Available: {list(self._PRESETS)}")
        # Real production schema presets (real table/column names, sample
        # values) do not ship in this public package -- only from an
        # institutional knowledge dir, same override as the knowledge base
        # above. Packaged copy is checked first for a generic/public
        # preset that happens to exist.
        from ledgerhound.config import get_str
        candidates = [Path(__file__).parent / "knowledge" / "defaults" / filename]
        override = get_str("knowledge", "knowledge_dir", env="LEDGERHOUND_KNOWLEDGE_DIR")
        if override:
            candidates.insert(0, Path(override) / filename)
        preset_path = next((p for p in candidates if p.exists()), None)
        if preset_path is None:
            raise FileNotFoundError(
                f"Preset file '{filename}' not found in "
                f"{[str(c) for c in candidates]}. Institutional presets "
                "are not shipped in this package -- set knowledge.knowledge_dir "
                "in main_config.yaml to a directory containing it.")
        with open(preset_path, encoding="utf-8") as f:
            data = json.load(f)
        return SchemaIngester().ingest(data)

    @property
    def eda(self):
        if self._eda is None:
            from ledgerhound.agents.eda_agent import EDAAgent
            self._eda = EDAAgent(
                llm=self._get_llm(), backend=self._backend,
                knowledge=self._knowledge)
        return self._eda

    def _get_llm(self):
        if self._llm is not None: return self._llm
        # If the router pinned the local LLM to a specific GPU, pass that
        # through so a HF model loads onto exactly that device instead of
        # device_map="auto" (which would shard across every GPU).
        pin = (self._compute.llm_device
               if self._compute and getattr(self._compute, "llm_device", "")
               else None)
        if self._llm_cfg_path:
            p = Path(self._llm_cfg_path)
            if not p.exists(): raise FileNotFoundError(f"LLM config not found: {p}")
            self._llm = _build_llm_from_config(p, llm_device=pin)
            return self._llm
        if self._llm_url:
            from ledgerhound.llm.client import LLMClient
            self._llm = LLMClient(base_url=self._llm_url,
                                   model=self._llm_name or "default",
                                   api_key=self._llm_key or "EMPTY")
            return self._llm
        raise ValueError("No LLM configured. Use --model or --llm-config.")

    @property
    def backend(self): return self._backend
    @property
    def schema(self): return self._schema
    @property
    def knowledge(self): return self._knowledge
    @property
    def compute(self): return self._compute


def _build_llm_from_config(config_path: Path, llm_device: str | None = None):
    import yaml as _yaml
    with open(config_path, encoding="utf-8") as f:
        data = _yaml.safe_load(f)
    cfg = dict(data.get("llm", data))
    provider = cfg.get("provider", "api")
    if provider == "huggingface_local":
        from ledgerhound.llm.hf_client import HFLocalClient
        # Honour the router's GPU pin, but never override an explicit,
        # non-"auto" device_map the user set in the config.
        if llm_device and str(cfg.get("device_map", "auto")).lower() == "auto":
            cfg["device_map"] = llm_device
        return HFLocalClient.from_dict(cfg)
    else:
        from ledgerhound.llm.client import LLMClient, LLMConfig
        return LLMClient(config=LLMConfig.from_dict(cfg))
