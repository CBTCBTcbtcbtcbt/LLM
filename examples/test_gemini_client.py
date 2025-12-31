import sys
import os
import yaml

# Add parent directory to path to import llm_client_gemini
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_client_gemini import LLMClientGemini

def load_config(config_path="config.yaml"):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file {config_path} not found. Using defaults.")
        return None

def main():
    # Try to load config
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
    config = load_config(config_path)
    
    if config and 'api' in config:
        api_config = config['api']
        api_key = api_config.get('api_key', 'YOUR_API_KEY')
        base_url = api_config.get('base_url', 'YOUR_BASE_URL')
        model = api_config.get('model', 'gemini-1.5-flash')
    else:
        api_key = "YOUR_API_KEY"
        base_url = "https://your-proxy-domain.com/v1beta"
        model = "gemini-1.5-flash"

    print("--- Testing Gemini Client ---")
    print(f"Base URL: {base_url}")
    print(f"Model: {model}")
    
    if api_key == "YOUR_API_KEY":
        print("\n[WARNING] Please set your API key in config.yaml or edit this script.")
        # Proceeding might fail
    
    try:
        client = LLMClientGemini(
            api_key=api_key,
            base_url=base_url,
            model=model
        )
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Please answer in a concise manner."},
            {"role": "user", "content": "Hello! Who are you?"}
        ]
        
        print("\nSending request (non-streaming)...")
        response = client.chat(messages)
        print(f"Response: {response}")
        
        print("\nSending request (streaming)...")
        print("Response: ", end="", flush=True)
        for chunk in client.stream_chat(messages):
            print(chunk, end="", flush=True)
        print("\n")
        
    except Exception as e:
        print(f"\n[ERROR] Request failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
