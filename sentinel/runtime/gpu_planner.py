"""
sentinel.runtime.gpu_planner
============================

Detect GPUs at session start, decide which role gets which device, expose
the plan to the rest of Sentinel.

Roles
-----
  cudf   : dataframe compute (read_parquet, groupby, etc.)
  llm    : local LLM inference (only when llm mode == "local")
  model  : modeling work (XGBoost, cuML, PyTorch fine-tunes, etc.)

Plan rules (LLM = API)
----------------------
  0 GPU   : cudf off, all CPU
  1 GPU   : GPU0 -> cudf + model  (no LLM contention; same device shared)
  2 GPU   : GPU0 -> cudf,  GPU1 -> model
  3+ GPU  : GPU0 -> cudf,  GPU1 -> model,  GPU2+ -> model pool

Plan rules (LLM = local)
------------------------
  0 GPU   : everything off; can't run local LLM
  1 GPU   : GPU0 -> llm  (cudf disabled -- pandas fallback)
  2 GPU   : GPU0 -> cudf,  GPU1 -> llm
  3 GPU   : GPU0 -> cudf,  GPU1 -> llm,   GPU2 -> model
  4+ GPU  : GPU0 -> cudf,  GPU1-2 -> llm, GPU3+ -> model pool

Persistence
-----------
Plan is computed once per session, serialised into ``session.json`` under
``gpu_plan``, and reused on resume. Re-detect only if the user runs
``sentinel start --replan-gpus``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict, field


@dataclass
class GPUInfo:
    idx: int
    name: str = "unknown"
    vram_total_gb: float = 0.0
    vram_free_gb: float = 0.0


@dataclass
class GPUPlan:
    """Role -> list of GPU indices."""
    cudf:  list[int] = field(default_factory=list)
    llm:   list[int] = field(default_factory=list)
    model: list[int] = field(default_factory=list)

    def cuda_visible(self, role: str) -> str:
        """Return a CUDA_VISIBLE_DEVICES string for the given role."""
        ids = getattr(self, role, [])
        return ",".join(str(i) for i in ids)

    def has(self, role: str) -> bool:
        return bool(getattr(self, role, []))


class GPUPlanner:
    """Detect GPUs and build a role-based allocation plan."""

    # Substrings that mean the LLM is an external API (no local GPU needed).
    _API_TOKENS = (
        "anthropic", "claude", "openai", "azure", "bedrock", "vertex",
        "gemini", "cohere", "together", "groq", "mistral.ai",
    )

    def __init__(self, llm_config: str = "", llm_is_local: bool | None = None):
        self.gpus: list[GPUInfo] = self._detect()
        self.llm_is_local = (
            llm_is_local
            if llm_is_local is not None
            else self._infer_llm_mode(llm_config)
        )
        self.plan: GPUPlan = self._build_plan()

    # -- Detection ---------------------------------------------------------

    @staticmethod
    def _detect() -> list[GPUInfo]:
        """Try pynvml; fall back to nvidia-smi parsing; else empty."""
        # Preferred: pynvml (ships with most CUDA installs)
        try:
            import pynvml
            pynvml.nvmlInit()
            out = []
            for i in range(pynvml.nvmlDeviceGetCount()):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                name = pynvml.nvmlDeviceGetName(h)
                if isinstance(name, bytes):
                    name = name.decode()
                out.append(GPUInfo(
                    idx=i, name=name,
                    vram_total_gb=round(mem.total / 1e9, 1),
                    vram_free_gb =round(mem.free  / 1e9, 1),
                ))
            pynvml.nvmlShutdown()
            return out
        except Exception:
            pass

        # Fallback: parse nvidia-smi if available
        try:
            import subprocess
            r = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=index,name,memory.total,memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode != 0:
                return []
            out = []
            for line in r.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    out.append(GPUInfo(
                        idx=int(parts[0]), name=parts[1],
                        vram_total_gb=round(int(parts[2]) / 1024, 1),
                        vram_free_gb =round(int(parts[3]) / 1024, 1),
                    ))
            return out
        except Exception:
            return []

    # -- LLM mode inference ------------------------------------------------

    @classmethod
    def _infer_llm_mode(cls, llm_config: str) -> bool:
        """True if the configured LLM runs locally on a GPU we own."""
        s = (llm_config or "").lower()
        if not s:
            return False  # default assume API -- safer for shared boxes
        if any(tok in s for tok in cls._API_TOKENS):
            return False
        # Heuristic: explicit local server signatures
        if any(tok in s for tok in ("vllm", "ollama", "tgi", "localhost",
                                    "127.0.0.1", "transformers", "llama.cpp")):
            return True
        return False

    # -- Plan construction -------------------------------------------------

    def _build_plan(self) -> GPUPlan:
        n = len(self.gpus)
        if n == 0:
            return GPUPlan()

        if self.llm_is_local:
            if n == 1:
                # GPU0 fully consumed by LLM; cudf disabled (pandas fallback)
                return GPUPlan(cudf=[], llm=[0], model=[])
            if n == 2:
                return GPUPlan(cudf=[0], llm=[1], model=[])
            if n == 3:
                return GPUPlan(cudf=[0], llm=[1], model=[2])
            # 4+
            return GPUPlan(cudf=[0], llm=[1, 2], model=list(range(3, n)))

        # LLM is API -- GPU 0 is free for compute
        if n == 1:
            # Single GPU shared between cudf and modeling (sequentially)
            return GPUPlan(cudf=[0], llm=[], model=[0])
        if n == 2:
            return GPUPlan(cudf=[0], llm=[], model=[1])
        # 3+
        return GPUPlan(cudf=[0], llm=[], model=list(range(1, n)))

    # -- Serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "gpus": [asdict(g) for g in self.gpus],
            "llm_is_local": self.llm_is_local,
            "plan": asdict(self.plan),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GPUPlanner":
        obj = cls.__new__(cls)
        obj.gpus = [GPUInfo(**g) for g in data.get("gpus", [])]
        obj.llm_is_local = data.get("llm_is_local", False)
        p = data.get("plan", {})
        obj.plan = GPUPlan(
            cudf=p.get("cudf", []),
            llm=p.get("llm", []),
            model=p.get("model", []),
        )
        return obj

    # -- Convenience -------------------------------------------------------

    def apply_env(self, role: str) -> None:
        """Set CUDA_VISIBLE_DEVICES for *role* in the current process."""
        os.environ["CUDA_VISIBLE_DEVICES"] = self.plan.cuda_visible(role)

    def use_cudf(self) -> bool:
        return self.plan.has("cudf")

    def summary(self) -> str:
        if not self.gpus:
            return "GPU Plan: no GPUs detected -- running on CPU (pandas)."
        lines = [f"GPU Plan ({'local LLM' if self.llm_is_local else 'API LLM'}):"]
        for g in self.gpus:
            lines.append(
                f"  [{g.idx}] {g.name}  "
                f"{g.vram_free_gb:.1f}/{g.vram_total_gb:.1f} GB free"
            )
        lines.append("")
        lines.append(f"  cudf  -> GPU {self.plan.cudf or 'OFF (pandas fallback)'}")
        if self.llm_is_local:
            lines.append(f"  llm   -> GPU {self.plan.llm   or 'OFF'}")
        lines.append(f"  model -> GPU {self.plan.model or 'CPU'}")
        return "\n".join(lines)
