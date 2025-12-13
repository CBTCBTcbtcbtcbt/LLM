"""Core LLM client module using OpenAI SDK."""
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from typing import List, Dict, Optional, Iterator, Iterable


class LLMClient:
    """Universal LLM client supporting OpenAI-compatible APIs."""
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_params = kwargs
    
    def chat(self, messages: Iterable[ChatCompletionMessageParam], **kwargs) -> str:
        """Send chat request and return response."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get('temperature', self.temperature),
            max_tokens=kwargs.get('max_tokens', self.max_tokens),
            **{k: v for k, v in self.extra_params.items() if k not in kwargs}
        )
        print(response)
        return response.choices[0].message.content or ""
    
    def stream_chat(self, messages: Iterable[ChatCompletionMessageParam], **kwargs) -> Iterator[str]:
        """Stream chat responses."""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get('temperature', self.temperature),
            max_tokens=kwargs.get('max_tokens', self.max_tokens),
            stream=True,
            **{k: v for k, v in self.extra_params.items() if k not in kwargs}
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
