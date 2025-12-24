"""
Cost Calculator
---------------
Calculate LLM request costs based on token usage and model pricing.

Prices are in USD per 1,000 tokens (as of 2024).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class ModelPricing:
    """Pricing information for a model."""

    input_per_1k: float  # Cost per 1K input tokens
    output_per_1k: float  # Cost per 1K output tokens


# Model pricing reference (USD per 1K tokens)
# Updated: December 2024
MODEL_PRICING: Dict[str, ModelPricing] = {
    # OpenAI
    "gpt-4o": ModelPricing(input_per_1k=0.005, output_per_1k=0.015),
    "gpt-4o-mini": ModelPricing(input_per_1k=0.00015, output_per_1k=0.0006),
    "gpt-4-turbo": ModelPricing(input_per_1k=0.01, output_per_1k=0.03),
    "gpt-3.5-turbo": ModelPricing(input_per_1k=0.0005, output_per_1k=0.0015),
    # Anthropic
    "claude-sonnet-4-20250514": ModelPricing(input_per_1k=0.003, output_per_1k=0.015),
    "claude-3-5-sonnet-20241022": ModelPricing(input_per_1k=0.003, output_per_1k=0.015),
    "claude-3-5-haiku-20241022": ModelPricing(input_per_1k=0.0008, output_per_1k=0.004),
    "claude-3-opus-20240229": ModelPricing(input_per_1k=0.015, output_per_1k=0.075),
    # Google Gemini
    "gemini-1.5-flash": ModelPricing(input_per_1k=0.000075, output_per_1k=0.0003),
    "gemini-1.5-pro": ModelPricing(input_per_1k=0.00125, output_per_1k=0.005),
    "gemini-2.0-flash": ModelPricing(input_per_1k=0.0001, output_per_1k=0.0004),
    # xAI Grok
    "grok-beta": ModelPricing(input_per_1k=0.005, output_per_1k=0.015),
    "grok-2-latest": ModelPricing(input_per_1k=0.005, output_per_1k=0.015),
}

# Fallback pricing for unknown models
DEFAULT_PRICING = ModelPricing(input_per_1k=0.01, output_per_1k=0.03)


class CostCalculator:
    """
    Calculate costs for LLM requests.

    Example:
        >>> calc = CostCalculator()
        >>> cost = calc.calculate("gpt-4o", tokens_input=500, tokens_output=200)
        >>> print(f"Cost: ${cost:.6f}")
    """

    def __init__(self, custom_pricing: Dict[str, ModelPricing] | None = None):
        """
        Initialize calculator with optional custom pricing.

        Args:
            custom_pricing: Override default pricing for specific models
        """
        self.pricing = {**MODEL_PRICING}
        if custom_pricing:
            self.pricing.update(custom_pricing)

    def calculate(
        self,
        model: str,
        tokens_total: int | None = None,
        tokens_input: int | None = None,
        tokens_output: int | None = None,
    ) -> float:
        """
        Calculate cost for a request.

        Args:
            model: Model name
            tokens_total: Total tokens (if input/output not separated)
            tokens_input: Input/prompt tokens
            tokens_output: Output/completion tokens

        Returns:
            Cost in USD

        Note:
            If only tokens_total is provided, assumes 30% input / 70% output split.
        """
        pricing = self.pricing.get(model, DEFAULT_PRICING)

        # Handle different input scenarios
        if tokens_input is not None and tokens_output is not None:
            input_cost = (tokens_input / 1000) * pricing.input_per_1k
            output_cost = (tokens_output / 1000) * pricing.output_per_1k
            return input_cost + output_cost

        elif tokens_total is not None:
            # Estimate split: typically ~30% input, ~70% output
            estimated_input = int(tokens_total * 0.3)
            estimated_output = int(tokens_total * 0.7)
            input_cost = (estimated_input / 1000) * pricing.input_per_1k
            output_cost = (estimated_output / 1000) * pricing.output_per_1k
            return input_cost + output_cost

        return 0.0

    def get_pricing(self, model: str) -> ModelPricing:
        """
        Get pricing for a specific model.

        Args:
            model: Model name

        Returns:
            ModelPricing dataclass
        """
        return self.pricing.get(model, DEFAULT_PRICING)

    def compare_models(
        self, models: list[str], tokens_input: int = 1000, tokens_output: int = 500
    ) -> Dict[str, float]:
        """
        Compare costs across multiple models.

        Args:
            models: List of model names to compare
            tokens_input: Input tokens for comparison
            tokens_output: Output tokens for comparison

        Returns:
            Dict mapping model name to cost

        Example:
            >>> calc.compare_models(["gpt-4o", "gpt-4o-mini", "claude-sonnet"])
            {'gpt-4o': 0.0125, 'gpt-4o-mini': 0.00045, 'claude-sonnet': 0.0105}
        """
        return {
            model: self.calculate(
                model, tokens_input=tokens_input, tokens_output=tokens_output
            )
            for model in models
        }

    @staticmethod
    def format_cost(cost: float) -> str:
        """
        Format cost for display.

        Args:
            cost: Cost in USD

        Returns:
            Formatted string like "$0.004500" or "<$0.01"
        """
        if cost < 0.01:
            return f"${cost:.6f}"
        return f"${cost:.2f}"
