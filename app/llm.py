"""Chat model factory for OpenAI-compatible LLM endpoints.

Supports OpenAI, Groq, Together, Ollama, vLLM, and any endpoint adhering
to the OpenAI chat completions protocol by reading base_url, api_key,
and model name from application settings.
"""

from typing import Any
from langchain_openai import ChatOpenAI
from app.config import get_settings


class LLMConfigurationError(RuntimeError):
    """Raised when required LLM configuration (e.g. API key) is missing."""
    pass


def get_chat_model(
    temperature: float = 0.0,
    model_name: str | None = None,
    **kwargs: Any,
) -> ChatOpenAI:
    """Create and return an initialized ChatOpenAI client.

    Args:
        temperature: Sampling temperature (0.0 for deterministic RAG answers).
        model_name: Optional model override; defaults to settings.openai_model.
        **kwargs: Extra parameters passed directly to ChatOpenAI.

    Returns:
        ChatOpenAI instance configured with settings.
    """
    settings = get_settings()
    api_key = settings.openai_api_key.strip() if settings.openai_api_key else ""

    if not api_key:
        raise LLMConfigurationError(
            "API key is not configured. Set OPENAI_API_KEY in your environment or .env file "
            "(works with OpenAI, Groq, or any OpenAI-compatible provider)."
        )

    return ChatOpenAI(
        api_key=api_key,
        base_url=settings.openai_api_base,
        model=model_name or settings.openai_model,
        temperature=temperature,
        **kwargs,
    )
