"""
Agent Coordinator
-----------------
Determines which specialized agent should handle each message.

Uses deterministic rule-based classification (0 tokens consumed)
for fast, predictable routing decisions.
"""

from __future__ import annotations

from typing import List, Optional

from llm_router.agents.base import AgentContext


class CoordinatorAgent:
    """
    Coordinator that routes messages to specialized agents.

    Uses keyword-based rules for instant classification without LLM calls.
    This approach provides:
    - 0 token cost for routing decisions
    - Sub-millisecond latency
    - 100% predictable behavior
    - Easy debugging and testing

    Example:
        >>> coordinator = CoordinatorAgent()
        >>> context = AgentContext(
        ...     messages=[{"role": "user", "content": "How much does it cost?"}]
        ... )
        >>> agent_type = coordinator.select_agent_kind(context)
        >>> print(agent_type)  # "SALES"
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize the coordinator.

        Args:
            config: Optional configuration for customizing behavior
        """
        self.config = config or {}

    # ─────────────────────────────────────────────
    # Classification Rules
    # ─────────────────────────────────────────────

    @staticmethod
    def _is_smalltalk(msg: str) -> bool:
        """
        Detect greetings, farewells, and courtesies.

        These are short, social messages that don't require
        specialized handling.
        """
        m = msg.strip().lower()
        if not m:
            return False

        # Short greetings and farewells (≤ 4 words)
        smalltalk_phrases = {
            # Greetings
            "hola",
            "buenas",
            "buen dia",
            "buen día",
            "buenos dias",
            "buenos días",
            "hi",
            "hey",
            "hello",
            "que tal",
            "qué tal",
            "como estas",
            "cómo estás",
            # Thanks
            "gracias",
            "muchas gracias",
            "thank you",
            "thanks",
            # Acknowledgments
            "ok",
            "ok!",
            "dale",
            "perfecto",
            "genial",
            "bueno",
            "entendido",
            "claro",
            "sí",
            "si",
            "no",
            # Farewells
            "chau",
            "adiós",
            "adios",
            "bye",
            "hasta luego",
            "nos vemos",
            "see you",
        }

        if len(m.split()) <= 4 and m in smalltalk_phrases:
            return True

        # Only emojis (no alphanumeric characters)
        if all(not ch.isalnum() for ch in m):
            return True

        return False

    @staticmethod
    def _is_sales(msg: str) -> bool:
        """
        Detect purchase intent, pricing questions, and upgrade requests.
        """
        text = msg.lower()
        keywords = [
            # Pricing
            "precio",
            "price",
            "plan",
            "planes",
            "plans",
            "cotización",
            "cotizar",
            "quote",
            "cuanto cuesta",
            "cuánto cuesta",
            "how much",
            "cuanto vale",
            "cuánto vale",
            "cost",
            "costo",
            "costos",
            "pricing",
            # Purchase
            "comprar",
            "buy",
            "purchase",
            "contratar",
            "subscribe",
            "subscription",
            "suscribir",
            "suscripción",
            "suscripcion",
            "probar",
            "demo",
            "trial",
            "prueba gratis",
            "free trial",
            # Comparison
            "vs",
            "versus",
            "diferencia entre",
            "difference between",
            "comparar",
            "compare",
            "cual es mejor",
            "which is better",
            # Upgrade / Limits
            "upgrade",
            "mejorar plan",
            "cambiar plan",
            "subir de plan",
            "más mensajes",
            "more messages",
            "sin crédito",
            "sin tokens",
            "out of credits",
            "límite",
            "limit",
            "alcanzado",
            # Features
            "incluye",
            "includes",
            "tiene",
            "has",
            "viene con",
            "comes with",
            "ofrece",
            "offers",
            "funcionalidades",
            "features",
            "características",
            # Business
            "licencia",
            "license",
            "factura",
            "invoice",
            "descuento",
            "discount",
            "oferta",
            "offer",
            # Payment
            "pagar",
            "pay",
            "pago",
            "payment",
            "forma de pago",
            "payment method",
            "tarjeta",
            "card",
            "transferencia",
            "transfer",
        ]
        return any(k in text for k in keywords)

    @staticmethod
    def _is_support(msg: str) -> bool:
        """
        Detect technical issues and help requests.
        """
        text = msg.lower()
        keywords = [
            # Problems
            "no funciona",
            "not working",
            "doesn't work",
            "no me anda",
            "no puedo",
            "can't",
            "cannot",
            "no logro",
            "error",
            "bug",
            "problema",
            "problem",
            "falla",
            "fallo",
            "fails",
            "broken",
            "no responde",
            "no carga",
            "not loading",
            # Help
            "ayuda",
            "help",
            "como hago",
            "cómo hago",
            "how do i",
            "necesito ayuda",
            "need help",
            "soporte",
            "support",
            "asistencia",
            "assistance",
            # Configuration
            "configurar",
            "configure",
            "setup",
            "instalar",
            "install",
            "conectar",
            "connect",
            "integrar",
            "integrate",
            "integración",
            "integration",
            # Access
            "no puedo entrar",
            "can't login",
            "can't access",
            "login",
            "contraseña",
            "password",
            "olvidé",
            "forgot",
            "recuperar",
            "recover",
            "reset",
            # Technical
            "api",
            "webhook",
            "widget",
            "dashboard",
            "analytics",
            "leads",
            "mensajes",
            "messages",
            # Urgency
            "urgente",
            "urgent",
            "rápido",
            "asap",
        ]
        return any(k in text for k in keywords)

    @staticmethod
    def _is_faq(msg: str) -> bool:
        """
        Detect general frequently asked questions.
        """
        text = msg.lower()
        keywords = [
            # Information
            "qué es",
            "que es",
            "what is",
            "como funciona",
            "cómo funciona",
            "how does",
            "para que sirve",
            "para qué sirve",
            "what for",
            # Capabilities
            "puede",
            "puedes",
            "can it",
            "can you",
            "sirve para",
            "used for",
            "hace",
            "does it",
            "permite",
            "allows",
            # Limits
            "límite",
            "limite",
            "limit",
            "cuanto",
            "cuánto",
            "how much",
            "how many",
            "máximo",
            "maximo",
            "maximum",
            "mínimo",
            "minimo",
            "minimum",
            # Location / Hours (for businesses)
            "horario",
            "horarios",
            "hours",
            "schedule",
            "ubicación",
            "location",
            "dirección",
            "address",
            "donde",
            "dónde",
            "where",
            "cuando",
            "cuándo",
            "when",
            "abierto",
            "open",
            "cerrado",
            "closed",
            # General
            "info",
            "información",
            "information",
            "detalles",
            "details",
            "explicame",
            "explícame",
            "explain",
        ]
        return any(k in text for k in keywords)

    # ─────────────────────────────────────────────
    # Agent Selection
    # ─────────────────────────────────────────────

    def select_agent_kind(
        self,
        context: AgentContext,
        allowed_agents: Optional[List[str]] = None,
    ) -> str:
        """
        Select the appropriate agent type for a message.

        Args:
            context: Agent context with conversation history
            allowed_agents: Optional list of allowed agent types
                           (for plan-based restrictions)

        Returns:
            Agent type: "SALES", "SUPPORT", "FAQ", "SMALLTALK", or "GENERAL"

        Example:
            >>> context = AgentContext(
            ...     messages=[{"role": "user", "content": "What's the price?"}]
            ... )
            >>> agent = coordinator.select_agent_kind(context)
            >>> print(agent)  # "SALES"
        """
        last_msg = context.last_user_message

        # 1. Deterministic classification (priority order)
        if self._is_smalltalk(last_msg):
            suggested = "SMALLTALK"
        elif self._is_sales(last_msg):
            suggested = "SALES"
        elif self._is_support(last_msg):
            suggested = "SUPPORT"
        elif self._is_faq(last_msg):
            suggested = "FAQ"
        else:
            suggested = "GENERAL"

        # 2. Plan-based gating
        if allowed_agents:
            allowed = {a.upper() for a in allowed_agents}
            if suggested not in allowed:
                # Fallback to GENERAL if available
                return "GENERAL" if "GENERAL" in allowed else "GENERAL"

        return suggested

    def get_agent_info(self, agent_kind: str) -> dict:
        """
        Get information about an agent type.

        Args:
            agent_kind: The agent type identifier

        Returns:
            Dict with agent metadata
        """
        info = {
            "SALES": {
                "description": "Handles pricing, plans, and purchase inquiries",
                "priority": "high",
                "typical_intents": ["pricing", "purchase", "upgrade"],
            },
            "SUPPORT": {
                "description": "Handles technical issues and help requests",
                "priority": "high",
                "typical_intents": ["error", "help", "configuration"],
            },
            "FAQ": {
                "description": "Handles frequently asked questions",
                "priority": "medium",
                "typical_intents": ["information", "how-to", "capabilities"],
            },
            "SMALLTALK": {
                "description": "Handles greetings and casual conversation",
                "priority": "low",
                "typical_intents": ["greeting", "farewell", "acknowledgment"],
            },
            "GENERAL": {
                "description": "Default handler for unclassified queries",
                "priority": "medium",
                "typical_intents": ["general", "unclassified"],
            },
        }
        return info.get(agent_kind, info["GENERAL"])
