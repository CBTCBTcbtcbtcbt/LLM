"""Conversation management module."""
from typing import List, Dict, Optional

try:
    from .llm_client import LLMClient
except ImportError:
    from llm_client import LLMClient


class Conversation:
    """Manages conversation history and context."""
    
    def __init__(self, client: LLMClient, system_prompt: Optional[str] = None):
        self.client = client
        self.messages: List[Dict[str, str]] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
    
    def add_message(self, role: str, content: str):
        """Add a message to conversation history."""
        self.messages.append({"role": role, "content": content})
    
    def send(self, user_message: str, **kwargs) -> str:
        """Send user message and get AI response."""
        self.add_message("user", user_message)
        response = self.client.chat(self.messages, **kwargs)
        self.add_message("assistant", response)
        return response
    
    def stream_send(self, user_message: str, **kwargs):
        """Send user message and stream AI response."""
        self.add_message("user", user_message)
        full_response = ""
        for chunk in self.client.stream_chat(self.messages, **kwargs):
            full_response += chunk
            yield chunk
        self.add_message("assistant", full_response)
    
    def clear(self):
        """Clear conversation history (keeps system prompt if exists)."""
        if self.messages and self.messages[0]["role"] == "system":
            self.messages = [self.messages[0]]
        else:
            self.messages = []
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.messages.copy()
    
    def set_system_prompt(self, prompt: str):
        """Set or update system prompt."""
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0]["content"] = prompt
        else:
            self.messages.insert(0, {"role": "system", "content": prompt})
