"""Core LLM client module supporting OpenAI and Google Gemini APIs."""
from typing import List, Dict, Optional, Iterator, Iterable, Union, Any
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
                        elif isinstance(item, dict) and item.get("__multimodal_file__"):
                            # Handle multimodal file dict (single or multiple)
                            from pathlib import Path
                            files_to_process = []
                            if "files" in item and isinstance(item["files"], list):
                                files_to_process = item["files"]
                            else:
                                files_to_process = [item]
                                
                            for file_info in files_to_process:
                                f_path = file_info.get("file_path")
                                f_mime = file_info.get("mime_type", "application/pdf")
                                f_name = file_info.get("filename", "document")
                                f_desc = file_info.get("description", f"Content of {f_name}")
                                
                                if f_path and Path(f_path).exists():
                                    try:
                                        f_bytes = Path(f_path).read_bytes()
                                        # Add descriptive text part first
                                        parts.append(types.Part(text=f_desc + ":"))
                                        # Add file content part
                                        parts.append(types.Part.from_bytes(data=f_bytes, mime_type=f_mime))
                                    except Exception:
                                        pass
                        else:
                            # Wrap supported objects (Image, File, etc.) in types.Part
                            try:
                                parts.append(types.Part(item))
                            except Exception:
                                # Fallback if direct wrapping fails, though SDK seems to support it
                                parts.append(item)
                elif isinstance(content, dict) and content.get("__multimodal_file__"):
                    # Handle multimodal file dict passed directly as content
                    from pathlib import Path
                    files_to_process = []
                    if "files" in content and isinstance(content["files"], list):
                        files_to_process = content["files"]
                    else:
                        files_to_process = [content]
                        
                    for file_info in files_to_process:
                        f_path = file_info.get("file_path")
                        f_mime = file_info.get("mime_type", "application/pdf")
                        f_name = file_info.get("filename", "document")
                        f_desc = file_info.get("description", f"Content of {f_name}")
                        
                        if f_path and Path(f_path).exists():
                            try:
                                f_bytes = Path(f_path).read_bytes()
                                # Add descriptive text part first
                                parts.append(types.Part(text=f_desc + ":"))
                                # Add file content part
                                parts.append(types.Part.from_bytes(data=f_bytes, mime_type=f_mime))
                            except Exception:
                                pass
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
        """Send chat request and return response.
        
        Args:
            messages: Chat messages in OpenAI format.
            **kwargs: Additional arguments including:
                - schema: Optional response schema for structured output (dict format).
                          If provided, response will be JSON; otherwise plain text.
                - temperature: Override default temperature.
                - max_tokens: Override default max tokens.
        """
        system_instruction, contents = self._convert_messages(messages)

        # Extract schema for structured output
        schema = kwargs.pop('schema', None)
        
        # Build config with optional structured output
        config_params = {
            "temperature": kwargs.get('temperature', self.temperature),
            "max_output_tokens": kwargs.get('max_tokens', self.max_tokens),
            "system_instruction": system_instruction,
        }
        
        # Add structured output parameters if schema is provided
        if schema is not None:
            config_params["response_schema"] = schema
            config_params["response_mime_type"] = "application/json"
        
        # Merge extra params
        for k, v in self.extra_params.items():
            if k not in kwargs and k not in config_params:
                config_params[k] = v
        
        config = types.GenerateContentConfig(**config_params)
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config
        )
        return response.text or ""
    
    def stream_chat(self, messages: Iterable[ChatCompletionMessageParam], **kwargs) -> Iterator[str]:
        """Stream chat responses.
        
        Args:
            messages: Chat messages in OpenAI format.
            **kwargs: Additional arguments including:
                - schema: Optional response schema for structured output (dict format).
                          If provided, response will be JSON; otherwise plain text.
                - temperature: Override default temperature.
                - max_tokens: Override default max tokens.
        """
        system_instruction, contents = self._convert_messages(messages)
        
        # Extract schema for structured output
        schema = kwargs.pop('schema', None)
        
        # Build config with optional structured output
        config_params = {
            "temperature": kwargs.get('temperature', self.temperature),
            "max_output_tokens": kwargs.get('max_tokens', self.max_tokens),
            "system_instruction": system_instruction,
        }
        
        # Add structured output parameters if schema is provided
        if schema is not None:
            config_params["response_schema"] = schema
            config_params["response_mime_type"] = "application/json"
        
        # Merge extra params
        for k, v in self.extra_params.items():
            if k not in kwargs and k not in config_params:
                config_params[k] = v
        
        config = types.GenerateContentConfig(**config_params)
        stream = self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config
        )
        
        for chunk in stream:
            if chunk.text:
                yield chunk.text

    def chat_with_tools(
        self, 
        messages: Iterable[ChatCompletionMessageParam],
        tools: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """Send chat request with function calling tools.
        
        Args:
            messages: Chat messages in OpenAI format.
            tools: List of function declarations. Each declaration should be a dict with:
                - name: Function name
                - description: Function description
                - parameters: JSON Schema object describing the function parameters
            **kwargs: Additional arguments including:
                - temperature: Override default temperature.
                - max_tokens: Override default max tokens.
        
        Returns:
            Dict with:
                - text: Response text (str or None)
                - function_call: Function call info (dict with 'name' and 'args') or None
                - model_response: Raw model response content for continuing conversation
                - contents: The contents sent to the model (for continuing conversation)
        """
        system_instruction, contents = self._convert_messages(messages)
        
        # Build tools configuration
        tool_declarations = types.Tool(function_declarations=tools)
        
        # Build config
        config_params = {
            "temperature": kwargs.get('temperature', self.temperature),
            "max_output_tokens": kwargs.get('max_tokens', self.max_tokens),
            "system_instruction": system_instruction,
            "tools": [tool_declarations],
        }
        
        # Merge extra params
        for k, v in self.extra_params.items():
            if k not in kwargs and k not in config_params:
                config_params[k] = v
        
        config = types.GenerateContentConfig(**config_params)
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config
        )
        
        # Extract result
        result = {
            "text": None,
            "function_call": None,
            "model_response": response.candidates[0].content if response.candidates else None,
            "contents": contents,
        }
        
        # Check for function call or text in response parts
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    result["function_call"] = {
                        "name": part.function_call.name,
                        "args": dict(part.function_call.args) if part.function_call.args else {}
                    }
                    break
                elif hasattr(part, 'text') and part.text:
                    result["text"] = part.text
        
        return result

    def continue_chat_with_tool_result(
        self,
        contents: List[types.Content],
        model_response: types.Content,
        function_name: str,
        function_result: Dict[str, Any],
        tools: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """Continue chat after function execution with the result.
        
        Args:
            contents: Previous conversation contents.
            model_response: The model's response containing the function call.
            function_name: Name of the executed function.
            function_result: Result from the function execution. Can be a regular dict
                           or a special multimodal file marker dict with '__multimodal_file__' key.
            tools: List of function declarations (same as chat_with_tools).
            **kwargs: Additional arguments.
        
        Returns:
            Dict with same structure as chat_with_tools.
        """
        from pathlib import Path
        
        # Append model response and function result to contents
        new_contents = list(contents)
        new_contents.append(model_response)
        
        # Check if function_result contains multimodal file marker
        user_parts = []
        
        if isinstance(function_result, dict) and function_result.get("__multimodal_file__"):
            # Handle multimodal file return (single or multiple)
            files_to_process = []
            if "files" in function_result and isinstance(function_result["files"], list):
                files_to_process = function_result["files"]
            else:
                files_to_process = [function_result]
            
            processed_descriptions = []
            
            successful_files = 0
            for file_info in files_to_process:
                file_path = file_info.get("file_path")
                mime_type = file_info.get("mime_type", "application/pdf")
                filename = file_info.get("filename", "document")
                description = file_info.get("description", f"File: {filename}")
                
                if file_path and Path(file_path).exists():
                    try:
                        # Read file bytes and create Part
                        file_bytes = Path(file_path).read_bytes()
                        
                        # Interleaved strategy: Description -> File
                        user_parts.append(types.Part(text=f"Content of {filename}:"))
                        
                        file_part = types.Part.from_bytes(
                            data=file_bytes,
                            mime_type=mime_type
                        )
                        user_parts.append(file_part)
                        
                        processed_descriptions.append(description)
                        successful_files += 1
                    except Exception as e:
                        processed_descriptions.append(f"Error reading {filename}: {str(e)}")
                else:
                    processed_descriptions.append(f"Error: File not found at {file_path}")
            
            # Create function response with summary
            result_summary = f"Processed {successful_files} files. " + "; ".join(processed_descriptions)
            function_response_part = types.Part.from_function_response(
                name=function_name,
                response={"result": result_summary, "file_count": successful_files},
            )
            
            # Prepend function response (it must be the first part or separate, but putting it first is safe)
            user_parts.insert(0, function_response_part)
        else:
            # Regular function result (text/dict)
            function_response_part = types.Part.from_function_response(
                name=function_name,
                response={"result": function_result},
            )
            user_parts.append(function_response_part)
        
        new_contents.append(types.Content(role="user", parts=user_parts))
        
        # Build tools configuration
        tool_declarations = types.Tool(function_declarations=tools)
        
        # Build config (no system_instruction needed as it's already in context)
        config_params = {
            "temperature": kwargs.get('temperature', self.temperature),
            "max_output_tokens": kwargs.get('max_tokens', self.max_tokens),
            "tools": [tool_declarations],
        }
        
        # Merge extra params
        for k, v in self.extra_params.items():
            if k not in kwargs and k not in config_params:
                config_params[k] = v
        
        config = types.GenerateContentConfig(**config_params)
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=new_contents,
            config=config
        )
        
        # Extract result
        result = {
            "text": None,
            "function_call": None,
            "model_response": response.candidates[0].content if response.candidates else None,
            "contents": new_contents,
        }
        
        # Check for function call or text in response parts
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    result["function_call"] = {
                        "name": part.function_call.name,
                        "args": dict(part.function_call.args) if part.function_call.args else {}
                    }
                    break
                elif hasattr(part, 'text') and part.text:
                    result["text"] = part.text
        
        return result

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
        """Send chat request and return response.
        
        Args:
            messages: Chat messages in OpenAI format.
            **kwargs: Additional arguments including:
                - schema: (Google provider only) Response schema for structured output.
                         Use google.genai.types.Schema to define the expected output format.
                - temperature: Override default temperature.
                - max_tokens: Override default max tokens.
        
        Returns:
            Response text from the model. If schema is provided (Google only), 
            returns JSON string that conforms to the schema.
        """
        return self.client_impl.chat(messages, **kwargs)
    
    def stream_chat(self, messages: Iterable[ChatCompletionMessageParam], **kwargs) -> Iterator[str]:
        """Stream chat responses.
        
        Args:
            messages: Chat messages in OpenAI format.
            **kwargs: Additional arguments including:
                - schema: (Google provider only) Response schema for structured output.
                         Use google.genai.types.Schema to define the expected output format.
                - temperature: Override default temperature.
                - max_tokens: Override default max tokens.
        
        Yields:
            Response text chunks from the model. If schema is provided (Google only),
            the complete response will be a JSON string that conforms to the schema.
        """
        return self.client_impl.stream_chat(messages, **kwargs)

    def chat_with_tools(
        self, 
        messages: Iterable[ChatCompletionMessageParam],
        tools: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """Send chat request with function calling tools (Google provider only).
        
        Args:
            messages: Chat messages in OpenAI format.
            tools: List of function declarations. Each declaration should be a dict with:
                - name: Function name
                - description: Function description
                - parameters: JSON Schema object describing the function parameters
            **kwargs: Additional arguments.
        
        Returns:
            Dict with:
                - text: Response text (str or None)
                - function_call: Function call info (dict with 'name' and 'args') or None
                - model_response: Raw model response content for continuing conversation
                - contents: The contents sent to the model (for continuing conversation)
        
        Raises:
            NotImplementedError: If provider is not 'google'.
        """
        if self.provider != "google":
            raise NotImplementedError("chat_with_tools is only supported for Google provider")
        return self.client_impl.chat_with_tools(messages, tools, **kwargs)

    def continue_chat_with_tool_result(
        self,
        contents: List,
        model_response: Any,
        function_name: str,
        function_result: Dict[str, Any],
        tools: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """Continue chat after function execution with the result (Google provider only).
        
        Args:
            contents: Previous conversation contents (from chat_with_tools response).
            model_response: The model's response containing the function call.
            function_name: Name of the executed function.
            function_result: Result from the function execution.
            tools: List of function declarations (same as chat_with_tools).
            **kwargs: Additional arguments.
        
        Returns:
            Dict with same structure as chat_with_tools.
        
        Raises:
            NotImplementedError: If provider is not 'google'.
        """
        if self.provider != "google":
            raise NotImplementedError("continue_chat_with_tool_result is only supported for Google provider")
        return self.client_impl.continue_chat_with_tool_result(
            contents, model_response, function_name, function_result, tools, **kwargs
        )
