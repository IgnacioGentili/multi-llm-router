"""
Gemini Provider
---------------
Implementation for Google's Gemini API.

Supports: gemini-1.5-flash, gemini-1.5-pro, gemini-2.0-flash
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

from llm_router.providers.base import LLMError


class GeminiProvider:
    """
    Provider for Google Gemini models.

    Note:
        Gemini uses a different message format internally.
        This provider automatically converts from OpenAI format.

    Example:
        >>> provider = GeminiProvider(model="gemini-1.5-flash")
        >>> response, tokens = provider.chat_completion(messages)
    """

    def __init__(self, model: str | None = None):
        """
        Initialize Gemini provider.

        Args:
            model: Model to use (default: gemini-1.5-flash)

        Raises:
            LLMError: If GOOGLE_API_KEY is not set
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise LLMError(
                message="GOOGLE_API_KEY environment variable is not set",
                provider="gemini",
            )

        self.model = model or os.getenv("DEFAULT_GEMINI_MODEL", "gemini-1.5-flash")

        try:
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            self._genai = genai
        except ImportError:
            raise LLMError(
                message="google-generativeai package not installed. "
                "Run: pip install google-generativeai",
                provider="gemini",
            )

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> Tuple[str, int]:
        """
        Call Google Gemini API.

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
            # Extract system instruction
            system_instruction = None
            gemini_messages = []

            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                if role == "system":
                    system_instruction = content
                elif role == "user":
                    gemini_messages.append({"role": "user", "parts": [content]})
                elif role == "assistant":
                    gemini_messages.append({"role": "model", "parts": [content]})

            # Configure model
            generation_config = {
                "temperature": temperature,
            }
            if max_tokens:
                generation_config["max_output_tokens"] = max_tokens

            model = self._genai.GenerativeModel(
                model_name=self.model,
                system_instruction=system_instruction,
                generation_config=generation_config,
            )

            # Start chat and send messages
            chat = model.start_chat(
                history=gemini_messages[:-1] if len(gemini_messages) > 1 else []
            )

            # Get last user message
            last_message = gemini_messages[-1]["parts"][0] if gemini_messages else ""
            response = chat.send_message(last_message)

            # Extract response
            reply = response.text if response.text else ""

            # Estimate tokens (Gemini doesn't always return token counts)
            # Using rough estimate: 1 token ≈ 4 characters
            tokens_estimate = (len(str(messages)) + len(reply)) // 4

            return reply, tokens_estimate

        except Exception as e:
            raise LLMError(
                message=f"API call failed: {str(e)}",
                provider="gemini",
                original_error=e,
            )
