"""Simple chat example."""
import sys
sys.path.append('..')

from llm_client import LLMClient
from conversation import Conversation
from chat import load_config

# Load API config and create client
config = load_config('../config.yaml')
api = config['api']

client = LLMClient(
    api_key=api['api_key'],
    base_url=api['base_url'],
    model=api['model'],
    temperature=api.get('temperature', 0.7),
    max_tokens=api.get('max_tokens', 2000)
)

# Create conversation
conv = Conversation(client, system_prompt="You are a helpful assistant.")

# Chat
response = conv.send("Hello! What can you help me with?")
print(f"AI: {response}")

response = conv.send("Tell me a joke")
print(f"AI: {response}")
