"""Agent example - creating AI agents with specific roles."""
import sys
sys.path.append('..')

from llm_client import create_client
from agent import Agent

# Create client
client = create_client(
    provider="openai",
    api_key="your-api-key-here",
    model="gpt-3.5-turbo",
    base_url="https://api.openai.com/v1"
)

# Create agents with different roles
teacher = Agent(
    name="Teacher",
    client=client,
    role="You are a patient teacher who explains concepts clearly.",
    personality="Friendly and encouraging"
)

scientist = Agent(
    name="Scientist",
    client=client,
    role="You are a scientist who provides technical and accurate information.",
    personality="Precise and analytical"
)

# Get responses from different agents
question = "What is photosynthesis?"

print("Teacher's response:")
print(teacher.respond(question))
print("\n" + "="*50 + "\n")

print("Scientist's response:")
print(scientist.respond(question))
