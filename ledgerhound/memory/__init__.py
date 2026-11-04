"""
ledgerhound.memory
===============

Dataset Memory -- the per-dataset L2 knowledge layer.

Each dataset Ledgerhound has ever seen gets a Dataset Memory file with:

* Role (fact / dimension / rollup)
* Schema columns (canonical names, semantic types, quality tags)
* Relationships (joins to other datasets)
* Rollup registry (named aggregated tables for fast access)
* **Sampling design** -- strata, methods, populations, encoded priors
* Findings log path (where its findings live)

Why per-dataset?
----------------
Because the same column means different things in different datasets,
and the same sample means different things under different sampling
designs. Cross-dataset learning is a Phase 3 problem; Phase 1 just
ensures each dataset's context survives across sessions.

Storage format: YAML for the design + JSONL for findings. Both human-
readable, both diff-able, both work without dependencies.
"""

from .models import (
    DatasetMemory,
    DatasetRole,
    SamplingDesign,
    SamplingMethod,
    SamplingStratum,
    DatasetRelationship,
    Rollup,
)
from .store import save_memory, load_memory
from .bootstrap import (
    bootstrap_from_data,
    bootstrap_from_yaml,
    infer_schema_columns,
    interview,
    list_datasets,
)
from .knowledge import DatasetKnowledge, KnowledgeEntry, KnowledgeType
from .session_log import SessionLog, SessionRecord
from .embeddings import Embedder, EmbedderInfo, cosine_similarity
from .vector_store import (
    VectorRecord,
    VectorStore,
    NumpyBruteForceStore,
    FaissCpuStore,
    create_store,
    load_store,
    save_store,
    available_backends,
)
from .longitudinal import (
    LongitudinalMemory,
    MemoryRecord,
    MemorySource,
    Recall,
    NAMESPACES,
)
from .fact_memory import (
    FactMemory,
    FactRecord,
    Observation,
    Consolidation,
)
from .lesson_memory import LessonMemory, Lesson
from .promotion_gate import (
    PromotionGate,
    PromotionCriteria,
    PromotionInput,
    PromotionDecision,
)
from .librarian import Librarian, RecallBundle, IngestReport, PromotionReport

__all__ = [
    "DatasetMemory",
    "DatasetRole",
    "SamplingDesign",
    "SamplingMethod",
    "SamplingStratum",
    "DatasetRelationship",
    "Rollup",
    "save_memory",
    "load_memory",
    "bootstrap_from_data",
    "bootstrap_from_yaml",
    "infer_schema_columns",
    "interview",
    "list_datasets",
    "DatasetKnowledge",
    "KnowledgeEntry",
    "KnowledgeType",
    "SessionLog",
    "SessionRecord",
    "Embedder",
    "EmbedderInfo",
    "cosine_similarity",
    "VectorRecord",
    "VectorStore",
    "NumpyBruteForceStore",
    "FaissCpuStore",
    "create_store",
    "load_store",
    "save_store",
    "available_backends",
    "LongitudinalMemory",
    "MemoryRecord",
    "MemorySource",
    "Recall",
    "NAMESPACES",
    "FactMemory",
    "FactRecord",
    "Observation",
    "Consolidation",
    "LessonMemory",
    "Lesson",
    "PromotionGate",
    "PromotionCriteria",
    "PromotionInput",
    "PromotionDecision",
    "Librarian",
    "RecallBundle",
    "IngestReport",
    "PromotionReport",
]
