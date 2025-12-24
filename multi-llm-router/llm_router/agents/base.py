"""
Agent Base Classes
------------------
Defines the context and protocol for all agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol


@dataclass
class AgentContext:
    """
    Canonical context that ALL agents receive.

    This ensures consistent information flow across different agent types.

    Attributes:
        messages: Conversation history in OpenAI format
        tenant_id: Tenant identifier (for multi-tenant systems)
        user_id: User identifier
        channel: Communication channel (web, whatsapp, telegram, etc.)
        config: Agent-specific configuration
        extra: Additional context data

    Example:
        >>> context = AgentContext(
        ...     messages=[{"role": "user", "content": "Hello"}],
        ...     channel="web"
        ... )
    """

    messages: List[Dict[str, str]]
    tenant_id: str | None = None
    user_id: str | None = None
    channel: str = "web"
    config: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def last_user_message(self) -> str:
        """Get the content of the last user message."""
        for msg in reversed(self.messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    @property
    def conversation_length(self) -> int:
        """Get the number of messages in the conversation."""
        return len(self.messages)


class BaseAgent(Protocol):
    """
    Protocol that all agents must implement.

    This defines the minimum interface for any agent in the system.

    Attributes:
        kind: Agent type identifier (SALES, SUPPORT, FAQ, etc.)
    """

    kind: str

    def build_messages(
        self, context: AgentContext, max_tokens: int = 4096
    ) -> List[Dict[str, str]]:
        """
        Build the final messages to send to the LLM.

        This method prepares the prompt but does NOT call the model.
        Separation of prompt building from execution allows for:
        - Testing prompts without API calls
        - Logging/debugging prompts
        - Using different models with the same prompt

        Args:
            context: The agent context with conversation and config
            max_tokens: Maximum tokens for the response

        Returns:
            List of messages ready for LLM consumption
        """
        ...


# Agent type constants
AGENT_TYPES = {
    "SALES": "Handles pricing, plans, purchases, and upgrades",
    "SUPPORT": "Handles technical issues and help requests",
    "FAQ": "Handles frequently asked questions",
    "SMALLTALK": "Handles greetings and casual conversation",
    "GENERAL": "Default agent for unclassified queries",
}
