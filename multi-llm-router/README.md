# 🚀 Multi-LLM Router

> Production-ready LLM orchestration system with smart routing, multi-agent coordination, and cost tracking.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 What is this?

A **production-tested** LLM orchestration layer that:

- **Routes requests** to the optimal model based on complexity
- **Supports multiple providers** (OpenAI, Anthropic, Google, xAI)
- **Coordinates specialized agents** (Sales, Support, FAQ, General)
- **Tracks costs** per request with detailed analytics

Built for real-world SaaS applications where you need to balance **quality, cost, and latency**.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Request                             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Agent Coordinator                           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Deterministic Classification (0 tokens)                 │    │
│  │  • Smalltalk detection (greetings, emojis)              │    │
│  │  • Sales intent (pricing, plans, upgrade)               │    │
│  │  • Support needs (errors, help, setup)                  │    │
│  │  • FAQ patterns (what is, how does)                     │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Smart Router                               │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Complexity Detection                                    │    │
│  │  • Low: "hola", "precio", "horario" → Fast model        │    │
│  │  • Medium: "explica", "describe" → Balanced model       │    │
│  │  • High: "analiza", "estrategia" → Premium model        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Strategies: cost_optimized | balanced | quality_optimized       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       LLM Factory                                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │  OpenAI  │ │ Anthropic│ │  Gemini  │ │   Grok   │           │
│  │ GPT-4o   │ │  Claude  │ │  Flash   │ │  Grok-2  │           │
│  │ GPT-4o-  │ │  Sonnet  │ │  Pro     │ │          │           │
│  │  mini    │ │          │ │          │ │          │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Cost Tracker                                │
│  • Per-request cost calculation                                  │
│  • Token usage tracking (input/output)                          │
│  • Provider-specific pricing                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

### 🔀 Smart Routing

Automatically selects the best model based on message complexity:

```python
from llm_router import SmartRouter

router = SmartRouter(strategy="balanced")

# Simple question → routes to gpt-4o-mini (cheap & fast)
provider, model = router.select_model(
    messages=[{"role": "user", "content": "Hola, cuál es el precio?"}]
)
# Returns: ("openai", "gpt-4o-mini")

# Complex analysis → routes to gpt-4o (premium)
provider, model = router.select_model(
    messages=[{"role": "user", "content": "Analiza esta estrategia de mercado y compara con competidores..."}]
)
# Returns: ("openai", "gpt-4o")
```

### 🏭 Provider Factory

Unified interface for multiple LLM providers:

```python
from llm_router import get_llm_provider

# OpenAI
openai = get_llm_provider("openai", model="gpt-4o")
response, tokens = openai.chat_completion(messages)

# Anthropic
claude = get_llm_provider("anthropic", model="claude-sonnet-4-20250514")
response, tokens = claude.chat_completion(messages)

# Gemini
gemini = get_llm_provider("gemini", model="gemini-1.5-flash")
response, tokens = gemini.chat_completion(messages)
```

### 🤖 Multi-Agent Coordination

Route messages to specialized agents without LLM calls:

```python
from llm_router.agents import CoordinatorAgent, AgentContext

coordinator = CoordinatorAgent()
context = AgentContext(
    messages=[{"role": "user", "content": "Cuánto cuesta el plan pro?"}]
)

agent_kind = coordinator.select_agent_kind(context)
# Returns: "SALES" (detected pricing intent)
```

### 💰 Cost Tracking

Track costs per request with provider-specific pricing:

```python
from llm_router import CostCalculator

calc = CostCalculator()

# Calculate cost for a request
cost = calc.calculate("gpt-4o", tokens_input=500, tokens_output=200)
print(f"Cost: ${cost:.6f}")  # Cost: $0.004500
```

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/multi-llm-router.git
cd multi-llm-router

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

---

## 🚀 Quick Start

```python
from llm_router import SmartRouter, get_llm_provider, CostCalculator

# 1. Initialize components
router = SmartRouter(strategy="balanced")
cost_calc = CostCalculator()

# 2. Get user message
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain how neural networks work"}
]

# 3. Smart route to best model
provider_name, model_name = router.select_model(messages=messages)
print(f"Routing to: {provider_name}/{model_name}")

# 4. Get provider and call LLM
provider = get_llm_provider(provider_name, model=model_name)
response, tokens_used = provider.chat_completion(messages)

# 5. Calculate cost
cost = cost_calc.calculate(model_name, tokens_used)
print(f"Response: {response[:100]}...")
print(f"Tokens: {tokens_used}, Cost: ${cost:.6f}")
```

---

## 📁 Project Structure

```
multi-llm-router/
├── README.md
├── requirements.txt
├── .env.example
├── main.py                      # Demo application
│
├── llm_router/
│   ├── __init__.py              # Public exports
│   ├── factory.py               # LLM provider factory
│   ├── router.py                # Smart routing logic
│   ├── cost.py                  # Cost calculation
│   │
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py              # Provider protocol
│   │   ├── openai_provider.py   # OpenAI implementation
│   │   ├── anthropic_provider.py # Anthropic implementation
│   │   └── gemini_provider.py   # Google Gemini implementation
│   │
│   └── agents/
│       ├── __init__.py
│       ├── base.py              # Agent context & protocol
│       └── coordinator.py       # Agent selection logic
│
└── examples/
    ├── basic_chat.py            # Simple chat example
    ├── smart_routing.py         # Routing demonstration
    └── multi_agent.py           # Agent coordination example
```

---

## ⚙️ Configuration

### Environment Variables

```bash
# Required: At least one provider
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...

# Optional: Additional providers
XAI_API_KEY=...

# Optional: Defaults
DEFAULT_PROVIDER=openai
DEFAULT_MODEL=gpt-4o-mini
ROUTING_STRATEGY=balanced  # cost_optimized | balanced | quality_optimized
```

### Routing Strategies

| Strategy            | Description                 | Best For                  |
| ------------------- | --------------------------- | ------------------------- |
| `cost_optimized`    | Always cheapest model       | High-volume, simple tasks |
| `balanced`          | Matches complexity to model | General use (recommended) |
| `quality_optimized` | Always best model           | Critical applications     |

---

## 💡 Design Decisions

### Why Deterministic Agent Classification?

The agent coordinator uses **keyword-based rules** instead of LLM classification:

- **0 tokens** consumed for routing decisions
- **<1ms latency** for classification
- **100% predictable** behavior
- **Easy to debug** and extend

For most applications, deterministic rules handle 90%+ of cases correctly.

### Why Factory Pattern for Providers?

- **Unified interface** across all providers
- **Lazy initialization** (only loads what you use)
- **Easy to add** new providers
- **Type-safe** with Protocol classes

### Why Smart Routing?

Not all queries need GPT-4o. By routing simple queries to cheaper models:

- **50-80% cost reduction** on typical workloads
- **Lower latency** for simple responses
- **Same quality** where it matters

---

## 📊 Cost Reference

| Model            | Input (1K tokens) | Output (1K tokens) |
| ---------------- | ----------------- | ------------------ |
| gpt-4o-mini      | $0.00015          | $0.0006            |
| gpt-4o           | $0.005            | $0.015             |
| claude-sonnet    | $0.003            | $0.015             |
| gemini-1.5-flash | $0.000075         | $0.0003            |
| gemini-1.5-pro   | $0.00125          | $0.005             |

---

## 🔧 Extending

### Adding a New Provider

```python
# llm_router/providers/new_provider.py
from .base import LLMProvider

class NewProvider(LLMProvider):
    model: str

    def __init__(self, model: str | None = None):
        self.model = model or "default-model"
        # Initialize client

    def chat_completion(self, messages, temperature=0.7, max_tokens=None):
        # Implement API call
        return response_text, tokens_used
```

### Adding Agent Types

```python
# In coordinator.py, add new detection method
@staticmethod
def _is_booking(msg: str) -> bool:
    keywords = ["reservar", "agendar", "cita", "turno", "appointment"]
    return any(k in msg.lower() for k in keywords)
```

---

## 📄 License

MIT License - feel free to use in your projects.

---

## 👤 Author

**Ignacio Gentili**

- LinkedIn: [/in/ignacio-gentili](https://linkedin.com/in/ignacio-gentili)
- GitHub: [@IgnacioGentili](https://github.com/IgnacioGentili)

---

## 🙏 Acknowledgments

This architecture powers [ÉTER](https://eter.network), a multi-tenant AI SaaS platform.
