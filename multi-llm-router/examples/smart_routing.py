#!/usr/bin/env python3
"""
Smart Routing Example
=====================

Demonstrates how the SmartRouter selects models based on query complexity.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_router import CostCalculator, SmartRouter


def main():
    # Initialize router with balanced strategy
    router = SmartRouter(strategy="balanced")
    cost_calc = CostCalculator()

    # Test different query complexities
    queries = [
        # Low complexity - will route to cheap model
        "Hi there!",
        "What's the price?",
        "Thanks!",
        # Medium complexity
        "How does your product work?",
        "What are the main features?",
        # High complexity - will route to premium model
        "Analyze the pros and cons of microservices vs monolithic architecture for a startup with 10 developers",
        "Create a detailed marketing strategy for launching a B2B SaaS product in the European market",
    ]

    print("Smart Routing Demo")
    print("=" * 70)
    print(f"Strategy: {router.strategy}")
    print("=" * 70)

    for query in queries:
        messages = [{"role": "user", "content": query}]

        # Get routing decision
        complexity = router.get_complexity(messages)
        provider, model = router.select_model(messages=messages)

        # Estimate cost for 1000 tokens
        estimated_cost = cost_calc.calculate(model, tokens_total=1000)

        print(f"\n📝 Query: \"{query[:50]}{'...' if len(query) > 50 else ''}\"")
        print(f"   Complexity: {complexity}")
        print(f"   Model: {provider}/{model}")
        print(f"   Est. cost (1K tokens): ${estimated_cost:.6f}")


if __name__ == "__main__":
    main()
