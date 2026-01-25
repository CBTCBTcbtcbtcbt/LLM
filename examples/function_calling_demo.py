"""
Function Calling Demo with Google Gemini

This example demonstrates how to use function calling (tool use) with the Agent class.
The agent can automatically call functions based on user requests.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_client import LLMClient
from agent import Agent


# ============================================================================
# Step 1: Define function declarations (tool definitions)
# ============================================================================

# Function declaration for controlling smart lights
set_light_values_declaration = {
    "name": "set_light_values",
    "description": "Sets the brightness and color temperature of a light.",
    "parameters": {
        "type": "object",
        "properties": {
            "brightness": {
                "type": "integer",
                "description": "Light level from 0 to 100. Zero is off and 100 is full brightness",
            },
            "color_temp": {
                "type": "string",
                "enum": ["daylight", "cool", "warm"],
                "description": "Color temperature of the light fixture, which can be `daylight`, `cool` or `warm`.",
            },
        },
        "required": ["brightness", "color_temp"],
    },
}

# Function declaration for getting weather
get_weather_declaration = {
    "name": "get_weather",
    "description": "Get the current weather for a location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city name, e.g., 'Beijing', 'New York'",
            },
        },
        "required": ["location"],
    },
}


# ============================================================================
# Step 2: Define actual function implementations
# ============================================================================

def set_light_values(brightness: int, color_temp: str) -> dict:
    """Set the brightness and color temperature of a room light (mock API).

    Args:
        brightness: Light level from 0 to 100.
        color_temp: Color temperature ('daylight', 'cool', or 'warm').

    Returns:
        A dictionary containing the set brightness and color temperature.
    """
    print(f"  [Function Called] set_light_values(brightness={brightness}, color_temp='{color_temp}')")
    return {"brightness": brightness, "colorTemperature": color_temp, "status": "success"}


def get_weather(location: str) -> dict:
    """Get weather for a location (mock API).

    Args:
        location: City name.

    Returns:
        Mock weather data.
    """
    print(f"  [Function Called] get_weather(location='{location}')")
    # Mock weather data
    weather_data = {
        "Beijing": {"temp": 15, "condition": "Sunny", "humidity": 45},
        "New York": {"temp": 8, "condition": "Cloudy", "humidity": 60},
        "Tokyo": {"temp": 12, "condition": "Rainy", "humidity": 75},
    }
    return weather_data.get(location, {"temp": 20, "condition": "Unknown", "humidity": 50})


# ============================================================================
# Main Demo
# ============================================================================

def main():
    # Get API key from environment
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: Please set GOOGLE_API_KEY or GEMINI_API_KEY environment variable")
        return

    # Create LLM client (Google provider)
    client = LLMClient(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com",
        model="gemini-2.0-flash",
        provider="google",
        temperature=0.7,
    )

    # ========================================================================
    # Demo 1: Auto-execute function calls
    # ========================================================================
    print("=" * 60)
    print("Demo 1: Auto-execute function calls")
    print("=" * 60)

    # Create agent with tools
    agent = Agent(
        client=client,
        name="SmartHomeAssistant",
        role="A helpful smart home assistant that can control lights and check weather.",
        tools=[set_light_values_declaration, get_weather_declaration],
        tool_handlers={
            "set_light_values": set_light_values,
            "get_weather": get_weather,
        }
    )

    # Test 1: Light control
    print("\n[User]: Turn the lights down to a romantic level")
    response = agent.respond("Turn the lights down to a romantic level")
    print(f"[Assistant]: {response}")

    # Reset for next test
    agent.reset()

    # Test 2: Weather query
    print("\n[User]: What's the weather like in Beijing?")
    response = agent.respond("What's the weather like in Beijing?")
    print(f"[Assistant]: {response}")

    # ========================================================================
    # Demo 2: Manual function execution
    # ========================================================================
    print("\n" + "=" * 60)
    print("Demo 2: Manual function execution (no auto-execute)")
    print("=" * 60)

    agent.reset()

    print("\n[User]: Set the lights to 80% brightness with warm color")
    
    # Get response without auto-execution
    response = agent.respond_with_tool_info("Set the lights to 80% brightness with warm color")
    
    if response.get("function_call"):
        func_call = response["function_call"]
        print(f"  [Model requested function call]: {func_call['name']}")
        print(f"  [Arguments]: {func_call['args']}")
        
        # Manually execute the function
        if func_call["name"] == "set_light_values":
            result = set_light_values(**func_call["args"])
            print(f"  [Function result]: {result}")
            
            # Continue conversation with result
            final_response = agent.execute_tool_and_continue(response, result)
            print(f"[Assistant]: {final_response.get('text', '')}")
    else:
        print(f"[Assistant]: {response.get('text', '')}")

    # ========================================================================
    # Demo 3: Using register_tool method
    # ========================================================================
    print("\n" + "=" * 60)
    print("Demo 3: Dynamic tool registration")
    print("=" * 60)

    # Create agent without tools
    agent2 = Agent(
        client=client,
        name="DynamicAssistant",
    )

    # Define a new tool
    calculator_declaration = {
        "name": "calculate",
        "description": "Perform basic arithmetic calculation.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate, e.g., '2 + 3 * 4'",
                },
            },
            "required": ["expression"],
        },
    }

    def calculate(expression: str) -> dict:
        """Evaluate a math expression (simple and safe)."""
        print(f"  [Function Called] calculate(expression='{expression}')")
        try:
            # Only allow safe characters for eval
            allowed_chars = set("0123456789+-*/.(). ")
            if not all(c in allowed_chars for c in expression):
                return {"error": "Invalid characters in expression"}
            result = eval(expression)
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}

    # Register tool dynamically
    agent2.register_tool(calculator_declaration, calculate)

    print("\n[User]: What is 15 * 7 + 23?")
    response = agent2.respond("What is 15 * 7 + 23?")
    print(f"[Assistant]: {response}")


if __name__ == "__main__":
    main()
