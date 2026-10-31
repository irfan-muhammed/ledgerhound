"""
sentinel.llm.hf_client
======================

Local HuggingFace model client -- loads a model directly onto GPU(s)
using ``transformers`` + ``bitsandbytes`` for 4-bit quantization.

Use this when:
  * The vLLM/SGLang API server is not running.
  * You want to test locally on a single GPU.
  * You have the model cached at ``HF_HOME``.

Usage::

    client = HFLocalClient(
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        hf_home="/data/.cache/huggingface/",
    )
    resp = client.generate([{"role": "user", "content": "Hello"}])

Or via config YAML::

    llm:
      provider: huggingface_local
      model: deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
      hf_home: /data/.cache/huggingface/
      quantize_4bit: true

The client implements the same ``generate()`` / ``health_check()``
interface as ``LLMClient``, so the EDA agent doesn't know or care
which backend is in use.
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class HFLocalConfig:
    """Configuration for a local HuggingFace model."""

    model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    hf_home: str = ""              # path to HF cache (sets HF_HOME env var)
    quantize_4bit: bool = True     # use bitsandbytes 4-bit quantization
    device_map: str = "auto"       # "auto", "cuda:0", etc.
    max_new_tokens: int = 2048
    temperature: float = 0.6
    do_sample: bool = False
    trust_remote_code: bool = False
    offline: bool = True           # set TRANSFORMERS_OFFLINE=1

    @classmethod
    def from_dict(cls, d: dict) -> HFLocalConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class HFLocalClient:
    """Local HuggingFace model with the same interface as LLMClient.

    Loads the model on first ``generate()`` call (lazy init) to avoid
    blocking import time.
    """

    def __init__(self, config: HFLocalConfig | None = None, **kw):
        self.config = config or HFLocalConfig(**{
            k: v for k, v in kw.items() if k in HFLocalConfig.__dataclass_fields__
        })
        self._model = None
        self._tokenizer = None
        self._pipe = None
        self._call_log: list[dict] = []

    @classmethod
    def from_dict(cls, d: dict) -> HFLocalClient:
        """Build from a config dict (e.g. parsed from YAML)."""
        return cls(config=HFLocalConfig.from_dict(d))

    @classmethod
    def from_config(cls, path: str) -> HFLocalClient:
        """Build from a YAML config file."""
        import yaml
        from pathlib import Path
        with open(Path(path), encoding="utf-8") as f:
            data = yaml.safe_load(f)
        cfg = data.get("llm", data)
        return cls.from_dict(cfg)

    # -- lazy model loading ---------------------------------

    def _load(self):
        """Load tokenizer + model on first use."""
        if self._pipe is not None:
            return

        cfg = self.config

        # Suppress noisy warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

        # Set environment
        if cfg.offline:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if cfg.hf_home:
            os.environ["HF_HOME"] = cfg.hf_home

        log.info("Loading model: %s (4bit=%s)", cfg.model, cfg.quantize_4bit)
        print(f"[Sentinel] Loading {cfg.model} ...")

        try:
            import torch
            from transformers import (
                AutoTokenizer, AutoModelForCausalLM,
                BitsAndBytesConfig, pipeline,
            )

            # Quantization config
            quant_config = None
            if cfg.quantize_4bit:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )

            self._tokenizer = AutoTokenizer.from_pretrained(cfg.model)
            self._model = AutoModelForCausalLM.from_pretrained(
                cfg.model,
                device_map=cfg.device_map,
                trust_remote_code=cfg.trust_remote_code,
                quantization_config=quant_config,
                use_cache=True,
            )

            self._pipe = pipeline(
                "text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=cfg.do_sample,
                pad_token_id=self._tokenizer.eos_token_id,
            )

            print(f"[Sentinel] OK Model loaded: {cfg.model}")

        except ImportError as e:
            raise ImportError(
                f"Missing dependency for local HF model: {e}\n"
                f"Install: pip install torch transformers bitsandbytes accelerate"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{cfg.model}': {e}\n"
                f"Check: model name correct? Enough GPU memory? "
                f"HF_HOME={cfg.hf_home or '(default)'}"
            ) from e

    # -- generate (same interface as LLMClient) -------------

    def generate(self, messages: list[dict], *, temperature: float | None = None,
                 max_tokens: int | None = None, **kw):
        """Generate a response from the local model.

        Accepts the same ``messages`` format as the API client::

            [{"role": "system", "content": "..."},
             {"role": "user",   "content": "..."}]
        """
        from sentinel.llm.client import LLMResponse

        self._load()

        # Build prompt from messages
        prompt = self._format_messages(messages)

        t0 = time.perf_counter()

        # Override generation params if provided
        gen_kwargs = {}
        if temperature is not None and temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["do_sample"] = True
        if max_tokens is not None:
            gen_kwargs["max_new_tokens"] = max_tokens

        outputs = self._pipe(prompt, **gen_kwargs)
        raw_text = outputs[0]["generated_text"]

        # Extract only the NEW text (after the prompt)
        if raw_text.startswith(prompt):
            content = raw_text[len(prompt):].strip()
        else:
            content = raw_text.strip()

        latency = (time.perf_counter() - t0) * 1000

        # Extract thinking blocks if present
        thinking = ""
        if "<think>" in content and "</think>" in content:
            i = content.index("<think>") + 7
            j = content.index("</think>")
            thinking = content[i:j].strip()
            content = content[j + 8:].strip()

        # Estimate tokens (rough: 1 token ~ 4 chars)
        prompt_tokens = len(prompt) // 4
        completion_tokens = len(content) // 4

        resp = LLMResponse(
            content=content,
            thinking=thinking,
            model=self.config.model,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            finish_reason="stop",
            latency_ms=round(latency, 1),
        )

        self._call_log.append({
            "ts": time.time(), "model": self.config.model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": resp.latency_ms,
            "has_thinking": resp.has_thinking,
        })

        return resp

    def _format_messages(self, messages: list[dict]) -> str:
        """Convert chat messages to a single prompt string.

        Uses the tokenizer's chat template if available,
        otherwise falls back to a simple format.
        """
        if self._tokenizer and hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass

        # Fallback: simple concatenation
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"[INST] {content} [/INST]")
            elif role == "user":
                parts.append(f"[INST] {content} [/INST]")
            elif role == "assistant":
                parts.append(content)
        return "\n".join(parts)

    # -- health check ---------------------------------------

    def health_check(self) -> dict:
        """Check if the model can be loaded."""
        try:
            self._load()
            return {"status": "healthy", "models": [self.config.model]}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @property
    def call_log(self) -> list[dict]:
        return self._call_log

    def close(self):
        """Release GPU memory."""
        if self._model is not None:
            del self._model
            del self._tokenizer
            del self._pipe
            self._model = None
            self._tokenizer = None
            self._pipe = None
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
