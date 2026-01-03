"""
Multimodal example using Google Gemini with Inline Data (Images).
Please set the API key and image path before running.
"""
import sys
import os
import yaml
from PIL import Image

# Add parent directory to path to import llm_client
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_client import LLMClient

def load_config(config_path="config.yaml"):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file {config_path} not found. Using defaults.")
        return None

def main():
    # Load config
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
    config = load_config(config_path)
    
    # Setup Client
    if config and 'api' in config:
        api_config = config['api']
        api_key = api_config.get('api_key', 'YOUR_API_KEY')
        base_url = api_config.get('base_url', 'YOUR_BASE_URL')
        model = api_config.get('model', 'gemini-1.5-flash')
    else:
        api_key = "YOUR_API_KEY"
        base_url = "https://generativelanguage.googleapis.com" # Default Google URL
        model = "gemini-1.5-flash"

    # PLACEHOLDER: Set your image path here
    
    image_path = "C:/Users/Mayn/Desktop/myfiles/project/2_303_20251226_1637.png"
    if not os.path.exists(image_path):
        print(f"[WARNING] Image file not found at: {image_path}")
        print("Please edit the script to point to a valid image file.")
        # Create a dummy image for testing logic flow if PIL is available
        try:
            print("Creating a dummy image for testing structure...")
            img = Image.new('RGB', (100, 100), color = 'red')
        except ImportError:
            print("PIL not installed. Please install pillow.")
            return
    else:
        print(f"Loading image from: {image_path}")
        img = Image.open(image_path)

    print(f"Initializing Client with provider='google'...")
    try:
        client = LLMClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            provider="google"
        )

        # Construct multimodal message
        # Content is a list of mixed text and image objects
        message_content = [
            "What is in this image?",
            img,
            "Describe it in detail."
        ]

        messages = [
            {"role": "user", "content": message_content}
        ]

        print("\nSending multimodal request...")
        response = client.chat(messages)
        print(f"Response: {response}")

    except Exception as e:
        print(f"\n[ERROR] Request failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
