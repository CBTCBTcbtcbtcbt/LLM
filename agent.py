"""Agent module for creating AI agents with specific roles."""
from typing import Optional, Dict, Any
from conversation import Conversation
from llm_client import LLMClient


class Agent:
    """Base class for AI agents with specific roles and behaviors."""
    
    def __init__(self, name: str, client: LLMClient, role: str = "", 
                 personality: str = ""):
        self.name = name
        self.client = client
        self.role = role
        self.personality = personality
        self.conversation = Conversation(
            client, 
            self._build_system_prompt()
        )
    
    def _build_system_prompt(self) -> str:
        """Build system prompt from role and personality."""
        prompt_parts = []
        if self.role:
            prompt_parts.append(f"Role: {self.role}")
        if self.personality:
            prompt_parts.append(f"Personality: {self.personality}")
        return "\n".join(prompt_parts) if prompt_parts else ""
    
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


class WerewolfPlayer(Agent):
    """Specialized agent for Werewolf game."""
    
    def __init__(self, name: str, client: LLMClient, character: str):
        self.character = character
        role = f"You are participating in the Werewolf game as {character}. Please gather information and make decisions based on your role."
        personality = self._get_character_personality(character)
        super().__init__(name, client, role, personality)
    
    def _get_character_personality(self, character: str) -> str:
        """Get personality traits based on character type."""
        personalities = {
            "werewolf": "You are deceptive and try to blend in with villagers.",
            "villager": "You are honest and try to find the werewolves.",
            "seer": "You can see one player's identity each night. Use this wisely.",
            "doctor": "You can protect one player each night from werewolf attacks.",
            "hunter": "If you die, you can take one player with you."
        }
        return personalities.get(character.lower(), "")
