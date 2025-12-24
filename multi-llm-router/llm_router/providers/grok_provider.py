"""
Grok Provider
-------------
Implementation for xAI's Grok API.

Supports: grok-beta, grok-2-latest
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

from llm_router.providers.base import LLMError


class GrokProvider:
    """
    Provider for xAI Grok models.

    Note:
        Grok uses OpenAI-compatible API with custom base URL.

    Example:
        >>> provider = GrokProvider(model="grok-2-latest")
        >>> response, tokens = provider.chat_completion(messages)
    """

    def __init__(self, model: str | None = None):
        """
        Initialize Grok provider.

        Args:
            model: Model to use (default: grok-2-latest)

        Raises:
            LLMError: If XAI_API_KEY is not set
        """
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise LLMError(
                message="XAI_API_KEY environment variable is not set", provider="grok"
            )

        self.model = model or os.getenv("DEFAULT_GROK_MODEL", "grok-2-latest")

        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        except ImportError:
            raise LLMError(
                message="openai package not installed. Run: pip install openai",
                provider="grok",
            )

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> Tuple[str, int]:
        """
        Call xAI Grok API.

        Args:
            messages: List of messages [{"role": "user", "content": "..."}]
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Tuple of (response_text, total_tokens)

        Raises:
            LLMError: If API call fails
        """
        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore
                temperature=temperature,
                max_tokens=max_tokens or 4096,
            )

            response = completion.choices[0].message.content or ""
            tokens = completion.usage.total_tokens if completion.usage else 0

            return response, tokens

        except Exception as e:
            raise LLMError(
                message=f"API call failed: {str(e)}", provider="grok", original_error=e
            )
