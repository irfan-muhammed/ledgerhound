"""LLM client layer -- works with any OpenAI-compatible endpoint or local HuggingFace model."""
from ledgerhound.llm.client import LLMClient, LLMConfig, LLMResponse

# HFLocalClient is imported lazily -- it requires torch/transformers
# which may not be installed.  Import it explicitly when needed:
#   from ledgerhound.llm.hf_client import HFLocalClient

__all__ = ["LLMClient", "LLMConfig", "LLMResponse"]
