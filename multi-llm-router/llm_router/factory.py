"""
LLM Factory
-----------
Factory pattern for creating LLM provider instances.

Supports lazy loading to only initialize providers when needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_router.providers.base import LLMProvider


def get_llm_provider(
    provider_name: str = "openai", model: str | None = None
) -> "LLMProvider":
    """
    Factory to get an LLM provider instance.

    Args:
        provider_name: Provider name (openai, anthropic, gemini, grok)
        model: Specific model to use (optional, uses provider default)

    Returns:
        LLMProvider instance ready for chat_completion calls

    Raises:
        ValueError: If provider is not supported

    Example:
        >>> provider = get_llm_provider("openai", model="gpt-4o")
        >>> response, tokens = provider.chat_completion(messages)
    """
    provider_name = provider_name.lower()

    if provider_name == "openai":
        from llm_router.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(model=model)

    elif provider_name == "anthropic":
        from llm_router.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(model=model)

    elif provider_name == "gemini":
        from llm_router.providers.gemini_provider import GeminiProvider

        return GeminiProvider(model=model)

    elif provider_name == "grok":
        from llm_router.providers.grok_provider import GrokProvider

        return GrokProvider(model=model)

    raise ValueError(
        f"Unsupported LLM provider: {provider_name}. "
        f"Supported: openai, anthropic, gemini, grok"
    )
