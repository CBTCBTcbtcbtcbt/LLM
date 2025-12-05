"""Core LLM client module for API interactions."""
import requests
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, api_key: str, model: str, **kwargs):
        self.api_key = api_key
        self.model = model
        self.config = kwargs
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send chat messages and get response."""
        pass
    
    @abstractmethod
    def stream_chat(self, messages: List[Dict[str, str]], **kwargs):
        """Stream chat responses."""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client."""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", 
                 base_url: str = "https://api.openai.com/v1", **kwargs):
        super().__init__(api_key, model, **kwargs)
        self.base_url = base_url.rstrip('/')
        self.temperature = kwargs.get('temperature', 0.7)
        self.max_tokens = kwargs.get('max_tokens', 2000)
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send chat request to OpenAI API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get('temperature', self.temperature),
            "max_tokens": kwargs.get('max_tokens', self.max_tokens)
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    
    def stream_chat(self, messages: List[Dict[str, str]], **kwargs):
        """Stream chat responses from OpenAI API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get('temperature', self.temperature),
            "max_tokens": kwargs.get('max_tokens', self.max_tokens),
            "stream": True
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data,
            stream=True
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    if line.strip() == 'data: [DONE]':
                        break
                    import json
                    try:
                        data = json.loads(line[6:])
                        if 'choices' in data and len(data['choices']) > 0:
                            delta = data['choices'][0].get('delta', {})
                            if 'content' in delta:
                                yield delta['content']
                    except json.JSONDecodeError:
                        continue


class AnthropicClient(LLMClient):
    """Anthropic Claude API client."""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229",
                 base_url: str = "https://api.anthropic.com", **kwargs):
        super().__init__(api_key, model, **kwargs)
        self.base_url = base_url.rstrip('/')
        self.temperature = kwargs.get('temperature', 0.7)
        self.max_tokens = kwargs.get('max_tokens', 2000)
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send chat request to Anthropic API."""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get('temperature', self.temperature),
            "max_tokens": kwargs.get('max_tokens', self.max_tokens)
        }
        
        response = requests.post(
            f"{self.base_url}/v1/messages",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        return response.json()['content'][0]['text']
    
    def stream_chat(self, messages: List[Dict[str, str]], **kwargs):
        """Stream chat responses from Anthropic API."""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get('temperature', self.temperature),
            "max_tokens": kwargs.get('max_tokens', self.max_tokens),
            "stream": True
        }
        
        response = requests.post(
            f"{self.base_url}/v1/messages",
            headers=headers,
            json=data,
            stream=True
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    import json
                    try:
                        data = json.loads(line[6:])
                        if data.get('type') == 'content_block_delta':
                            yield data['delta'].get('text', '')
                    except json.JSONDecodeError:
                        continue


def create_client(provider: str, **config) -> LLMClient:
    """Factory function to create LLM client based on provider."""
    clients = {
        'openai': OpenAIClient,
        'anthropic': AnthropicClient
    }
    
    if provider not in clients:
        raise ValueError(f"Unsupported provider: {provider}")
    
    return clients[provider](**config)
