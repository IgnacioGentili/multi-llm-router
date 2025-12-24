#!/usr/bin/env python3
"""
Multi-LLM Router - Demo Application
===================================

This script demonstrates the key features of the multi-llm-router library.

Usage:
    python main.py

Requirements:
    - Set OPENAI_API_KEY (or other provider keys) in .env file
    - pip install -r requirements.txt
"""

import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def demo_smart_routing():
    """Demonstrate smart routing based on message complexity."""
    from llm_router import SmartRouter

    print("\n" + "=" * 60)
    print("🔀 SMART ROUTING DEMO")
    print("=" * 60)

    router = SmartRouter(strategy="balanced")

    test_messages = [
        # Low complexity
        [{"role": "user", "content": "Hola"}],
        [{"role": "user", "content": "What's the price?"}],
        # Medium complexity
        [{"role": "user", "content": "Explain how your service works"}],
        [{"role": "user", "content": "What are the differences between the plans?"}],
        # High complexity
        [
            {
                "role": "user",
                "content": "Analyze our current marketing strategy and compare it with industry best practices for B2B SaaS companies",
            }
        ],
        [
            {
                "role": "user",
                "content": "I need a detailed financial projection for the next 5 years considering market trends",
            }
        ],
    ]

    for messages in test_messages:
        content = messages[0]["content"]
        complexity = router.get_complexity(messages)
        provider, model = router.select_model(messages=messages)

        print(f"\n📝 Message: \"{content[:50]}{'...' if len(content) > 50 else ''}\"")
        print(f"   Complexity: {complexity}")
        print(f"   → Routes to: {provider}/{model}")


def demo_agent_coordination():
    """Demonstrate multi-agent routing."""
    from llm_router.agents import AgentContext, CoordinatorAgent

    print("\n" + "=" * 60)
    print("🤖 AGENT COORDINATION DEMO")
    print("=" * 60)

    coordinator = CoordinatorAgent()

    test_queries = [
        "Hola, buenos días!",  # SMALLTALK
        "How much does the pro plan cost?",  # SALES
        "The widget is not loading",  # SUPPORT
        "What features are included?",  # FAQ
        "Tell me about quantum physics",  # GENERAL
        "I want to upgrade my account",  # SALES
        "Help me configure the integration",  # SUPPORT
    ]

    for query in test_queries:
        context = AgentContext(messages=[{"role": "user", "content": query}])
        agent_type = coordinator.select_agent_kind(context)

        print(f'\n📝 "{query}"')
        print(f"   → Agent: {agent_type}")


def demo_cost_calculation():
    """Demonstrate cost calculation."""
    from llm_router import CostCalculator

    print("\n" + "=" * 60)
    print("💰 COST CALCULATION DEMO")
    print("=" * 60)

    calc = CostCalculator()

    # Compare models for same workload
    models = ["gpt-4o", "gpt-4o-mini", "claude-sonnet-4-20250514", "gemini-1.5-flash"]
    tokens_input = 1000
    tokens_output = 500

    print(
        f"\nCost comparison for {tokens_input} input + {tokens_output} output tokens:\n"
    )

    costs = calc.compare_models(models, tokens_input, tokens_output)

    # Sort by cost
    sorted_costs = sorted(costs.items(), key=lambda x: x[1])

    for model, cost in sorted_costs:
        print(f"  {model:30} {CostCalculator.format_cost(cost):>12}")

    # Calculate savings
    most_expensive = sorted_costs[-1][1]
    cheapest = sorted_costs[0][1]
    savings = ((most_expensive - cheapest) / most_expensive) * 100

    print(f"\n  💡 Savings with smart routing: up to {savings:.0f}%")


def demo_full_flow():
    """Demonstrate the complete flow with actual API call."""
    from llm_router import CostCalculator, SmartRouter, get_llm_provider

    print("\n" + "=" * 60)
    print("🚀 FULL FLOW DEMO (with API call)")
    print("=" * 60)

    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  OPENAI_API_KEY not set. Skipping live API demo.")
        print("   Set it in .env file to see this demo.")
        return

    router = SmartRouter(strategy="balanced")
    cost_calc = CostCalculator()

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": "What is Python in one sentence?"},
    ]

    print(f"\n📝 Query: \"{messages[-1]['content']}\"")

    # 1. Route to best model
    provider_name, model_name = router.select_model(messages=messages)
    print(f"🔀 Routing to: {provider_name}/{model_name}")

    # 2. Get provider and call
    try:
        provider = get_llm_provider(provider_name, model=model_name)
        response, tokens_used = provider.chat_completion(messages, max_tokens=100)

        # 3. Calculate cost
        cost = cost_calc.calculate(model_name, tokens_total=tokens_used)

        print(f"\n✅ Response: {response}")
        print(f"📊 Tokens used: {tokens_used}")
        print(f"💰 Cost: {CostCalculator.format_cost(cost)}")

    except Exception as e:
        print(f"\n❌ Error: {e}")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("   MULTI-LLM ROUTER - DEMONSTRATION")
    print("=" * 60)

    demo_smart_routing()
    demo_agent_coordination()
    demo_cost_calculation()
    demo_full_flow()

    print("\n" + "=" * 60)
    print("✅ Demo complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
