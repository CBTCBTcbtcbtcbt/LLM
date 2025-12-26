"""Main chat interface for interacting with LLM."""
import yaml
from llm_client import LLMClient
from conversation import Conversation


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    """Main chat interface."""
    config = load_config()
    api_config = config['api']
    
    client = LLMClient(
        api_key=api_config['api_key'],
        base_url=api_config['base_url'],
        model=api_config['model'],
        temperature=api_config.get('temperature', 0.7),
        max_tokens=api_config.get('max_tokens', 2000)
    )
    
    conversation = Conversation(client)
    
    print("Chat started! Type 'quit' to exit, 'clear' to reset conversation.")
    print("-" * 50)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        if user_input.lower() == 'clear':
            conversation.clear()
            print("Conversation cleared.")
            continue
        
        try:
            print("\nAI: ", end="", flush=True)
            for chunk in conversation.stream_send(user_input):
                print(chunk, end="", flush=True)
            print()
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
