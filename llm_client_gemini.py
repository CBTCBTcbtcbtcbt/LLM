"""Core LLM client module using Google GenAI SDK (Gemini)."""
from google import genai
from google.genai import types
from openai.types.chat import ChatCompletionMessageParam
from typing import Iterator, Iterable, Optional, List

class LLMClientGemini:
    """Universal LLM client supporting Google GenAI (Gemini) API."""
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ):
        # Extract api_version if present in kwargs, otherwise it might be in extra_params
        api_version = kwargs.pop('api_version', None)
        
        # Configure client with http_options for custom base_url
        http_options_args = {"base_url": base_url}
        if api_version:
            http_options_args["api_version"] = api_version

        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(**http_options_args)
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_params = kwargs
    
    def _convert_messages(self, messages: Iterable[ChatCompletionMessageParam]) -> tuple[Optional[str], List[types.Content]]:
        """Convert OpenAI format messages to Gemini format."""
        system_instruction = None
        contents = []
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            
            # Skip empty content
            if not content:
                continue
                
            content_str = str(content)
            
            if role == "system":
                if system_instruction is None:
                    system_instruction = content_str
                else:
                    system_instruction += "\n" + content_str
            elif role == "user":
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part(text=content_str)]
                ))
            elif role == "assistant":
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part(text=content_str)]
                ))
                
        return system_instruction, contents

    def chat(self, messages: Iterable[ChatCompletionMessageParam], **kwargs) -> str:
        """Send chat request and return response."""
        system_instruction, contents = self._convert_messages(messages)
        
        config = types.GenerateContentConfig(
            temperature=kwargs.get('temperature', self.temperature),
            max_output_tokens=kwargs.get('max_tokens', self.max_tokens),
            system_instruction=system_instruction,
            **{k: v for k, v in self.extra_params.items() if k not in kwargs}
        )
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config
        )
        return response.text or ""
    
    def stream_chat(self, messages: Iterable[ChatCompletionMessageParam], **kwargs) -> Iterator[str]:
        """Stream chat responses."""
        system_instruction, contents = self._convert_messages(messages)
        
        config = types.GenerateContentConfig(
            temperature=kwargs.get('temperature', self.temperature),
            max_output_tokens=kwargs.get('max_tokens', self.max_tokens),
            system_instruction=system_instruction,
            **{k: v for k, v in self.extra_params.items() if k not in kwargs}
        )
        
        stream = self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config
        )
        
        for chunk in stream:
            if chunk.text:
                yield chunk.text
