"""
Anthropic Provider
------------------
Implementation for Anthropic's Claude API.

Supports: claude-sonnet-4, claude-3.5-sonnet, claude-3-opus, claude-3-haiku
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

from llm_router.providers.base import LLMError


class AnthropicProvider:
    """
    Provider for Anthropic Claude models.

    Note:
        Anthropic's API handles system prompts differently.
        This provider automatically separates system messages.

    Example:
        >>> provider = AnthropicProvider(model="claude-sonnet-4-20250514")
        >>> response, tokens = provider.chat_completion(messages)
    """

    def __init__(self, model: str | None = None):
        """
        Initialize Anthropic provider.

        Args:
            model: Model to use (default: claude-sonnet-4-20250514)

        Raises:
            LLMError: If ANTHROPIC_API_KEY is not set
        """
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMError(
                message="ANTHROPIC_API_KEY environment variable is not set",
                provider="anthropic",
            )

        self.model = model or os.getenv(
            "DEFAULT_ANTHROPIC_MODEL", "claude-sonnet-4-20250514"
        )

        try:
            from anthropic import Anthropic

            self._client = Anthropic(api_key=api_key)
        except ImportError:
            raise LLMError(
                message="anthropic package not installed. Run: pip install anthropic",
                provider="anthropic",
            )

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> Tuple[str, int]:
        """
        Call Anthropic Claude API.

        Args:
            messages: List of messages [{"role": "user", "content": "..."}]
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Tuple of (response_text, total_tokens)

        Raises:
            LLMError: If API call fails

        Note:
            System messages are automatically extracted and passed
            via the `system` parameter as required by Anthropic's API.
        """
        try:
            # Separate system prompt from conversation messages
            # Anthropic requires system prompts to be passed separately
            system_prompt = ""
            conversation_messages = []

            for msg in messages:
                if msg.get("role") == "system":
                    system_prompt = msg.get("content", "")
                else:
                    conversation_messages.append(msg)

            # Build request
            request_kwargs = {
                "model": self.model,
                "messages": conversation_messages,
                "temperature": temperature,
                "max_tokens": max_tokens or 4096,
            }

            # Only add system if present
            if system_prompt:
                request_kwargs["system"] = system_prompt

            response = self._client.messages.create(**request_kwargs)

            # Extract response text
            reply = response.content[0].text if response.content else ""

            # Calculate total tokens
            tokens = response.usage.input_tokens + response.usage.output_tokens

            return reply, tokens

        except Exception as e:
            raise LLMError(
                message=f"API call failed: {str(e)}",
                provider="anthropic",
                original_error=e,
            )
