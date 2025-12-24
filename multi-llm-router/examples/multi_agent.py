#!/usr/bin/env python3
"""
Multi-Agent Example
===================

Demonstrates how the CoordinatorAgent routes messages to specialized agents.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_router.agents import AgentContext, CoordinatorAgent


def main():
    # Initialize coordinator
    coordinator = CoordinatorAgent()

    # Test messages for different intents
    test_cases = [
        # Smalltalk
        ("Hola!", "SMALLTALK"),
        ("Thanks for your help!", "SMALLTALK"),
        ("👍", "SMALLTALK"),
        # Sales
        ("How much does it cost?", "SALES"),
        ("I want to upgrade to pro", "SALES"),
        ("What's included in the premium plan?", "SALES"),
        # Support
        ("The app is not working", "SUPPORT"),
        ("I can't login to my account", "SUPPORT"),
        ("Help me configure the API", "SUPPORT"),
        # FAQ
        ("What is your product?", "FAQ"),
        ("How does it work?", "FAQ"),
        ("What are your business hours?", "FAQ"),
        # General
        ("Tell me a joke", "GENERAL"),
        ("What do you think about AI?", "GENERAL"),
    ]

    print("Multi-Agent Coordination Demo")
    print("=" * 70)

    correct = 0
    total = len(test_cases)

    for message, expected in test_cases:
        context = AgentContext(messages=[{"role": "user", "content": message}])

        result = coordinator.select_agent_kind(context)
        status = "✅" if result == expected else "❌"

        if result == expected:
            correct += 1

        print(f'\n{status} "{message}"')
        print(f"   Expected: {expected}")
        print(f"   Got: {result}")

    print("\n" + "=" * 70)
    print(f"Accuracy: {correct}/{total} ({100*correct/total:.0f}%)")

    # Show agent info
    print("\n\nAgent Types Available:")
    print("-" * 40)
    for agent_type in ["SALES", "SUPPORT", "FAQ", "SMALLTALK", "GENERAL"]:
        info = coordinator.get_agent_info(agent_type)
        print(f"\n{agent_type}:")
        print(f"  {info['description']}")
        print(f"  Priority: {info['priority']}")


if __name__ == "__main__":
    main()
