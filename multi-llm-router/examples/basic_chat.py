#!/usr/bin/env python3
"""
Basic Chat Example
==================

Simple example showing how to use the LLM factory for chat completion.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
from llm_router import get_llm_provider

load_dotenv()


def main():
    # Initialize provider
    provider = get_llm_provider("openai", model="gpt-4o-mini")

    # Build messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    # Get completion
    response, tokens = provider.chat_completion(
        messages=messages, temperature=0.7, max_tokens=100
    )

    print(f"Response: {response}")
    print(f"Tokens used: {tokens}")


if __name__ == "__main__":
    main()
