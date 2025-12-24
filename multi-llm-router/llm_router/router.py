"""
Smart Router
------------
Intelligent model selection based on message complexity.

Routes simple queries to cheap/fast models and complex queries
to premium models, optimizing cost without sacrificing quality.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from llm_router.cost import CostCalculator

# Keywords that indicate message complexity
COMPLEXITY_KEYWORDS = {
    "high": [
        # Analysis & Strategy
        "analyze",
        "analiza",
        "compare",
        "compara",
        "evaluate",
        "evalúa",
        "strategy",
        "estrategia",
        "plan",
        "planifica",
        "recommend",
        "recomienda",
        "diagnostic",
        "diagnóstico",
        "complex",
        "complejo",
        "detailed",
        "detallado",
        # Professional domains
        "contract",
        "contrato",
        "legal",
        "financial",
        "financiero",
        "technical",
        "técnico",
        "architecture",
        "arquitectura",
    ],
    "medium": [
        # Explanation & Description
        "explain",
        "explica",
        "describe",
        "describes",
        "summarize",
        "resume",
        "list",
        "lista",
        "what is",
        "qué es",
        "how does",
        "cómo funciona",
        "difference",
        "diferencia",
        "advantages",
        "ventajas",
        "pros and cons",
    ],
    "low": [
        # Greetings & Simple queries
        "hi",
        "hola",
        "hello",
        "hey",
        "thanks",
        "gracias",
        "yes",
        "sí",
        "no",
        "ok",
        "dale",
        "sure",
        # Quick info
        "price",
        "precio",
        "hours",
        "horario",
        "location",
        "ubicación",
        "contact",
        "contacto",
        "address",
        "dirección",
    ],
}


class SmartRouter:
    """
    Smart router that selects the optimal model based on context.

    Strategies:
        - cost_optimized: Always use cheapest model
        - quality_optimized: Always use best model
        - balanced: Match model to query complexity (default)

    Example:
        >>> router = SmartRouter(strategy="balanced")
        >>> provider, model = router.select_model(messages=messages)
        >>> print(f"Using {provider}/{model}")
    """

    def __init__(self, strategy: str = "balanced"):
        """
        Initialize the smart router.

        Args:
            strategy: Routing strategy (cost_optimized, quality_optimized, balanced)
        """
        if strategy not in ("cost_optimized", "quality_optimized", "balanced"):
            raise ValueError(f"Invalid strategy: {strategy}")

        self.strategy = strategy
        self._cost_calculator = CostCalculator()

    def select_model(
        self,
        messages: List[Dict[str, Any]],
        config: Dict[str, Any] | None = None,
    ) -> Tuple[str, str]:
        """
        Select the best model for the given messages.

        Args:
            messages: Conversation messages in OpenAI format
            config: Optional configuration overrides
                - force_provider: Force specific provider
                - force_model: Force specific model
                - preferred_provider: Preferred provider if no force

        Returns:
            Tuple of (provider_name, model_name)

        Example:
            >>> messages = [{"role": "user", "content": "What's the price?"}]
            >>> provider, model = router.select_model(messages)
            >>> # Returns: ("openai", "gpt-4o-mini") for simple query
        """
        config = config or {}

        # 1. Check for forced model
        if config.get("force_model"):
            provider = config.get("force_provider", "openai")
            model = config["force_model"]
            return provider, model

        # 2. Detect complexity
        complexity = self._detect_complexity(messages)

        # 3. Select based on strategy
        if self.strategy == "cost_optimized":
            return self._cheapest_model()

        elif self.strategy == "quality_optimized":
            return self._best_model()

        else:  # balanced
            preferred_provider = config.get("preferred_provider", "openai")
            return self._balanced_model(complexity, preferred_provider)

    def _detect_complexity(self, messages: List[Dict[str, Any]]) -> str:
        """
        Detect message complexity based on keywords and length.

        Returns:
            "low", "medium", or "high"
        """
        if not messages:
            return "low"

        # Get last user message
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                # Handle string or list content
                if isinstance(content, str):
                    last_user_msg = content.lower()
                elif isinstance(content, list):
                    # Multi-modal: extract text parts
                    texts = [
                        p.get("text", "") for p in content if p.get("type") == "text"
                    ]
                    last_user_msg = " ".join(texts).lower()
                break

        # Check keywords (high priority first)
        for keyword in COMPLEXITY_KEYWORDS["high"]:
            if keyword in last_user_msg:
                return "high"

        for keyword in COMPLEXITY_KEYWORDS["medium"]:
            if keyword in last_user_msg:
                return "medium"

        # Length-based detection
        if len(last_user_msg) > 200:
            return "high"
        elif len(last_user_msg) > 50:
            return "medium"

        return "low"

    def _cheapest_model(self) -> Tuple[str, str]:
        """Return the cheapest available model."""
        return "gemini", "gemini-1.5-flash"

    def _best_model(self) -> Tuple[str, str]:
        """Return the highest quality model."""
        return "anthropic", "claude-sonnet-4-20250514"

    def _balanced_model(
        self, complexity: str, preferred_provider: str = "openai"
    ) -> Tuple[str, str]:
        """
        Select model based on complexity, respecting provider preference.

        Args:
            complexity: Detected complexity level
            preferred_provider: Provider to prefer when possible

        Returns:
            Tuple of (provider, model)
        """
        if complexity == "low":
            # Fast, cheap models for simple queries
            model_map = {
                "openai": ("openai", "gpt-4o-mini"),
                "anthropic": ("anthropic", "claude-3-5-haiku-20241022"),
                "gemini": ("gemini", "gemini-1.5-flash"),
                "grok": ("grok", "grok-beta"),
            }
            return model_map.get(preferred_provider, ("openai", "gpt-4o-mini"))

        elif complexity == "medium":
            # Balanced models
            model_map = {
                "openai": ("openai", "gpt-4o-mini"),
                "anthropic": ("anthropic", "claude-3-5-sonnet-20241022"),
                "gemini": ("gemini", "gemini-1.5-flash"),
                "grok": ("grok", "grok-2-latest"),
            }
            return model_map.get(preferred_provider, ("openai", "gpt-4o-mini"))

        else:  # high
            # Premium models for complex queries
            model_map = {
                "openai": ("openai", "gpt-4o"),
                "anthropic": ("anthropic", "claude-sonnet-4-20250514"),
                "gemini": ("gemini", "gemini-1.5-pro"),
                "grok": ("grok", "grok-2-latest"),
            }
            return model_map.get(preferred_provider, ("openai", "gpt-4o"))

    def estimate_cost(self, model: str, tokens: int) -> float:
        """
        Estimate cost for a request.

        Args:
            model: Model name
            tokens: Token count

        Returns:
            Estimated cost in USD
        """
        return self._cost_calculator.calculate(model, tokens)

    def get_complexity(self, messages: List[Dict[str, Any]]) -> str:
        """
        Public method to get complexity for debugging.

        Args:
            messages: Conversation messages

        Returns:
            Complexity level: "low", "medium", or "high"
        """
        return self._detect_complexity(messages)
