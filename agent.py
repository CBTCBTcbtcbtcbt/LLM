"""Agent module for creating AI agents with specific roles."""
from typing import Optional
from conversation import Conversation
from llm_client import LLMClient


class Agent:
    """AI agent with specific role and personality."""
    
    def __init__(
        self,
        client: LLMClient,
        name: str = "Agent",
        role: str = "",
        personality: str = "",
        background: str = "",
    ):
        self.client = client
        self.name = name
        self.role = role
        self.personality = personality
        self.background = background
        self.conversation = Conversation(client, self._build_system_prompt())
    
    def _build_system_prompt(self) -> str:
        """Build system prompt from role and personality."""
        parts = []
        parts.append( """
        # Role-Playing Instructions
        **【CORE RULES: STRICT PROHIBITION AGAINST BREAKING CHARACTER】**
        1.  **Identity Constraint:** Throughout the entire conversation, you are **ABSOLUTELY FORBIDDEN** to reveal or imply that you are an Artificial Intelligence, a Large Language Model, a program, or any non-human entity. You are 【Character's Name/Identity】.
        2.  **Response Requirement:** All your answers must be **STRICTLY** given from the 'I' first-person perspective of your character. The tone and content must align with your personality and the established scenario.
        3.  **Interaction Style:** Focus on 【E.g., Dialogue Exchange / Situational Description】. Do not provide any systematic summaries or analyses that break character.
        """)
        if self.role:
            parts.append(f"""
                        **【ROLE DEFINITION】**
                        You are: {self.role}
                        """)
        if self.personality:
            parts.append(f"""**Personality Traits:**
                          {self.personality}
                        """)
        if self.background:
            parts.append(f"""**【BACKGROUND INFORMATION·】**
                          {self.background}
                        """)
        return "\n".join(parts) if parts else ""
    
    def respond(self, message: str, **kwargs) -> str:
        """Generate response to a message."""
        return self.conversation.send(message, **kwargs)
    
    def stream_respond(self, message: str, **kwargs):
        """Stream response to a message."""
        yield from self.conversation.stream_send(message, **kwargs)
    
    def reset(self):
        """Reset agent's conversation history."""
        self.conversation.clear()
    
    def get_history(self):
        """Get agent's conversation history."""
        return self.conversation.get_history()
