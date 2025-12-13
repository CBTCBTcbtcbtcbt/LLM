"""Simple chat example."""
import sys
import os
import json
sys.path.append('..')

from llm_client import LLMClient
from conversation import Conversation
from chat import load_config 

prompt_path = "prompt.json"

# Load API config and create client
config = load_config('../config.yaml')
api = config['api']

with open(prompt_path, 'r', encoding='utf-8') as f:
        # 使用 json.load() 读取文件对象 f 中的 JSON 数据
        prompt = json.load(f)

client = LLMClient(
    api_key=api['api_key'],
    base_url=api['base_url'],
    model=api['model'],
    temperature=api.get('temperature', 0.7),
    max_tokens=api.get('max_tokens', 10000)
)

# Create conversation
conv = Conversation(client, system_prompt=prompt['system_prompt'])

# Chat
response = conv.send(user_message=prompt['user_prompt'])
print(f"AI: {response}")


