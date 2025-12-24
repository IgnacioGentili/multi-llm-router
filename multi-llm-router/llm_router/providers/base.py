"""
Base Provider Protocol
----------------------
Defines the interface that all LLM providers must implement.

Uses Python's Protocol for structural subtyping (duck typing with type hints).
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, Tuple


class LLMProvider(Protocol):
    """
    Protocol that all LLM providers must implement.

    This enables type hints without concrete coupling,
    following the Dependency Inversion Principle.

    Attributes:
        model: The model identifier being used

    Example:
        >>> def call_llm(provider: LLMProvider, messages: list) -> str:
        ...     response, tokens = provider.chat_completion(messages)
        ...     return response
    """

    model: str

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> Tuple[str, int]:
        """
        Execute a chat completion request.

        Args:
            messages: List of messages in OpenAI format:
                [{"role": "system", "content": "..."},
                 {"role": "user", "content": "..."},
                 {"role": "assistant", "content": "..."}]
            temperature: Sampling temperature (0.0 - 2.0)
            max_tokens: Maximum tokens to generate (None = model default)

        Returns:
            Tuple of (response_text, tokens_used)

        Raises:
            LLMError: If the API call fails
        """
        ...


class LLMError(Exception):
    """
    Exception raised when an LLM API call fails.

    Attributes:
        message: Error description
        provider: Provider that raised the error
        original_error: Original exception if any
    """

    def __init__(
        self, message: str, provider: str, original_error: Exception | None = None
    ):
        self.message = message
        self.provider = provider
        self.original_error = original_error
        super().__init__(f"[{provider}] {message}")
