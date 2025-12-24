"""
OpenAI Provider
---------------
Implementation for OpenAI's chat completion API.

Supports: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

from llm_router.providers.base import LLMError


class OpenAIProvider:
    """
    Provider for OpenAI models.

    Example:
        >>> provider = OpenAIProvider(model="gpt-4o")
        >>> response, tokens = provider.chat_completion(messages)
    """

    def __init__(self, model: str | None = None):
        """
        Initialize OpenAI provider.

        Args:
            model: Model to use (default: gpt-4o-mini)

        Raises:
            LLMError: If OPENAI_API_KEY is not set
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMError(
                message="OPENAI_API_KEY environment variable is not set",
                provider="openai",
            )

        self.model = model or os.getenv("DEFAULT_OPENAI_MODEL", "gpt-4o-mini")

        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=api_key)
        except ImportError:
            raise LLMError(
                message="openai package not installed. Run: pip install openai",
                provider="openai",
            )

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> Tuple[str, int]:
        """
        Call OpenAI chat completion API.

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
                max_tokens=max_tokens,
            )

            response = completion.choices[0].message.content or ""
            tokens = completion.usage.total_tokens if completion.usage else 0

            return response, tokens

        except Exception as e:
            raise LLMError(
                message=f"API call failed: {str(e)}",
                provider="openai",
                original_error=e,
            )
