"""Simple chat example."""
import sys
sys.path.append('..')

from llm_client import create_client
from conversation import Conversation
from chat import load_config

# Load API config and create client
_config = load_config()
_api = _config.get('api', {})
client = create_client(
    provider=_api.get('provider', 'openai'),
    api_key=_api.get('api_key', ''),
    model=_api.get('model', 'gpt-3.5-turbo'),
    base_url=_api.get('base_url', 'https://api.openai.com/v1'),
    temperature=_api.get('temperature', 0.7),
    max_tokens=_api.get('max_tokens', 2000)
)

# Create conversation
conv = Conversation(client, system_prompt="You are a helpful assistant.")

# Chat
response = conv.send("Hello! What can you help me with?")
print(f"AI: {response}")

response = conv.send("Tell me a joke")
print(f"AI: {response}")
