"""Core LLM client module supporting OpenAI and Google Gemini APIs."""
from typing import List, Dict, Optional, Iterator, Iterable, Union
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

try:
    from google import genai
    from google.genai import types
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

class _OpenAIClient:
    """Internal OpenAI client implementation."""
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ):
        if base_url and not base_url.endswith('/v1'):
            base_url = f"{base_url.rstrip('/')}/v1"
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

class _GeminiClient:
    """Internal Google Gemini client implementation."""
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ):
        if not HAS_GEMINI:
            raise ImportError("Google GenAI SDK is not installed. Please install 'google-genai'.")
            
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
            
            if role == "system":
                # Handle system instruction
                if isinstance(content, str):
                    content_str = content
                elif isinstance(content, list):
                    content_str = "\n".join(str(item) for item in content if isinstance(item, str))
                else:
                    content_str = str(content)
                    
                if system_instruction is None:
                    system_instruction = content_str
                else:
                    system_instruction += "\n" + content_str
            
            elif role in ["user", "assistant"]:
                parts = []
                if isinstance(content, str):
                    parts.append(types.Part(text=content))
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, str):
                            parts.append(types.Part(text=item))
                        else:
                            # Wrap supported objects (Image, File, etc.) in types.Part
                            try:
                                parts.append(types.Part(item))
                            except Exception:
                                # Fallback if direct wrapping fails, though SDK seems to support it
                                parts.append(item)
                else:
                    # Fallback for other types (convert to string)
                    parts.append(types.Part(text=str(content)))
                
                mapped_role = "user" if role == "user" else "model"
                contents.append(types.Content(
                    role=mapped_role,
                    parts=parts
                ))
                
        return system_instruction, contents

    def chat(self, messages: Iterable[ChatCompletionMessageParam], **kwargs) -> str:
        """Send chat request and return response."""
        system_instruction, contents = self._convert_messages(messages)

        # Extract structured output parameters
        response_mime_type = kwargs.pop('response_mime_type', None)
        response_schema = kwargs.pop('response_schema', None)
        
        config = types.GenerateContentConfig(
            temperature=kwargs.get('temperature', self.temperature),
            max_output_tokens=kwargs.get('max_tokens', self.max_tokens),
            system_instruction=system_instruction,
            response_mime_type=response_mime_type,
            response_schema=response_schema,
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
        
        # Extract structured output parameters
        response_mime_type = kwargs.pop('response_mime_type', None)
        response_schema = kwargs.pop('response_schema', None)

        config = types.GenerateContentConfig(
            temperature=kwargs.get('temperature', self.temperature),
            max_output_tokens=kwargs.get('max_tokens', self.max_tokens),
            system_instruction=system_instruction,
            response_mime_type=response_mime_type,
            response_schema=response_schema,
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

class LLMClient:
    """Universal LLM client supporting OpenAI and Google Gemini APIs."""
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        provider: str = "openai",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ):
        """
        Initialize the LLM client.
        
        Args:
            api_key: API key for the provider.
            base_url: Base URL for the API.
            model: Model name to use.
            provider: "openai" or "google". Defaults to "openai".
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional provider-specific arguments.
        """
        self.provider = provider.lower()
        if self.provider == "google":
            self.client_impl = _GeminiClient(
                api_key=api_key,
                base_url=base_url,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        else:
            self.client_impl = _OpenAIClient(
                api_key=api_key,
                base_url=base_url,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
    
    def chat(self, messages: Iterable[ChatCompletionMessageParam], **kwargs) -> str:
        """Send chat request and return response."""
        return self.client_impl.chat(messages, **kwargs)
    
    def stream_chat(self, messages: Iterable[ChatCompletionMessageParam], **kwargs) -> Iterator[str]:
        """Stream chat responses."""
        return self.client_impl.stream_chat(messages, **kwargs)
