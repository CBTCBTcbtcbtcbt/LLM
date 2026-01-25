"""Agent module for creating AI agents with specific roles."""
from typing import Optional, List, Dict, Any, Callable
from .conversation import Conversation
from .llm_client import LLMClient


class Agent:
    """AI agent with specific role and personality, supporting function calling."""
    
    def __init__(
        self,
        client: LLMClient,
        name: str = "Agent",
        role: str = "",
        personality: str = "",
        background: str = "",
        tools: List[Dict[str, Any]] = None,
        tool_handlers: Dict[str, Callable] = None,
    ):
        """
        Initialize the Agent.
        
        Args:
            client: LLM client instance.
            name: Agent name.
            role: Agent role description.
            personality: Agent personality traits.
            background: Agent background information.
            tools: List of function declarations for function calling. Each declaration
                   should be a dict with 'name', 'description', and 'parameters'.
            tool_handlers: Dict mapping function names to actual callable functions.
                          e.g., {"set_light_values": set_light_values_func}
        """
        self.client = client
        self.name = name
        self.role = role
        self.personality = personality
        self.background = background
        self.tools = tools or []
        self.tool_handlers = tool_handlers or {}
        self.conversation = Conversation(client, self._build_system_prompt())
    
    def _build_system_prompt(self) -> str:
        parts = []
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
    
    def register_tool(self, declaration: Dict[str, Any], handler: Callable):
        """Register a tool (function) for the agent to use.
        
        Args:
            declaration: Function declaration dict with 'name', 'description', 'parameters'.
            handler: The actual callable function to execute.
        """
        self.tools.append(declaration)
        self.tool_handlers[declaration["name"]] = handler
    
    def respond(self, message: str, auto_execute_tools: bool = True, max_tool_calls: int = 5, **kwargs) -> str:
        """Generate response to a message, with optional automatic function calling.
        
        Args:
            message: User message.
            auto_execute_tools: If True, automatically execute function calls and continue
                               conversation until a text response is received.
            max_tool_calls: Maximum number of consecutive tool calls to prevent infinite loops.
            **kwargs: Additional arguments passed to the LLM client.
        
        Returns:
            Final text response from the model.
        """
        # If no tools registered, use normal conversation
        if not self.tools:
            return self.conversation.send(message, **kwargs)
        
        # Add user message to history
        self.conversation.add_message("user", message)
        
        # Use chat_with_tools
        response = self.client.chat_with_tools(
            self.conversation.messages,
            self.tools,
            **kwargs
        )
        
        tool_call_count = 0
        
        # Loop while there are function calls to handle
        while response.get("function_call") and auto_execute_tools and tool_call_count < max_tool_calls:
            function_call = response["function_call"]
            func_name = function_call["name"]
            func_args = function_call["args"]
            
            # Execute the function if handler exists
            if func_name in self.tool_handlers:
                try:
                    func_result = self.tool_handlers[func_name](**func_args)
                except Exception as e:
                    func_result = {"error": str(e)}
            else:
                func_result = {"error": f"Unknown function: {func_name}"}
            
            # Continue conversation with function result
            response = self.client.continue_chat_with_tool_result(
                contents=response["contents"],
                model_response=response["model_response"],
                function_name=func_name,
                function_result=func_result,
                tools=self.tools,
                **kwargs
            )
            
            tool_call_count += 1
        
        # Get final text response
        final_text = response.get("text", "")
        
        # Add assistant response to history
        if final_text:
            self.conversation.add_message("assistant", final_text)
        
        return final_text or ""
    
    def respond_with_tool_info(self, message: str, **kwargs) -> Dict[str, Any]:
        """Generate response and return full information including any function calls.
        
        This method does NOT auto-execute functions. It returns the raw response
        so the caller can decide how to handle function calls.
        
        Args:
            message: User message.
            **kwargs: Additional arguments passed to the LLM client.
        
        Returns:
            Dict with:
                - text: Response text (str or None)
                - function_call: Function call info (dict with 'name' and 'args') or None
                - model_response: Raw model response for continuing conversation
                - contents: Conversation contents for continuing conversation
        """
        if not self.tools:
            # No tools, use normal chat
            response_text = self.conversation.send(message, **kwargs)
            return {"text": response_text, "function_call": None}
        
        self.conversation.add_message("user", message)
        
        response = self.client.chat_with_tools(
            self.conversation.messages,
            self.tools,
            **kwargs
        )
        
        # If text response, add to history
        if response.get("text"):
            self.conversation.add_message("assistant", response["text"])
        
        return response
    
    def execute_tool_and_continue(
        self, 
        prev_response: Dict[str, Any],
        function_result: Any,
        **kwargs
    ) -> Dict[str, Any]:
        """Continue conversation after manually executing a function.
        
        Use this after respond_with_tool_info() when you want to manually
        execute the function and continue the conversation.
        
        Args:
            prev_response: The response dict from respond_with_tool_info().
            function_result: The result from executing the function.
            **kwargs: Additional arguments.
        
        Returns:
            Dict with same structure as respond_with_tool_info().
        """
        if not prev_response.get("function_call"):
            return prev_response
        
        func_name = prev_response["function_call"]["name"]
        
        response = self.client.continue_chat_with_tool_result(
            contents=prev_response["contents"],
            model_response=prev_response["model_response"],
            function_name=func_name,
            function_result=function_result,
            tools=self.tools,
            **kwargs
        )
        
        # If text response, add to history
        if response.get("text"):
            self.conversation.add_message("assistant", response["text"])
        
        return response
    
    def stream_respond(self, message: str, **kwargs):
        """Stream response to a message.
        
        Note: Function calling is not supported in streaming mode.
        """
        yield from self.conversation.stream_send(message, **kwargs)
    
    def reset(self):
        """Reset agent's conversation history."""
        self.conversation.clear()
    
    def get_history(self):
        """Get agent's conversation history."""
        return self.conversation.get_history()
