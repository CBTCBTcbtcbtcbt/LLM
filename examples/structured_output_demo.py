"""
Structured Output Demo for Google Gemini Provider

This example demonstrates how to use the schema parameter for structured output
when using the Google provider. The schema ensures the model returns data in a
specific JSON format.

Note: This feature only works with provider="google".

Schema Format: Use simple dict/YAML format - no need to import google.genai.types!
"""
import json
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_client import LLMClient


def demo_basic_structured_output():
    """Basic structured output example - extracting person info."""
    print("=" * 60)
    print("Demo 1: Basic Structured Output (Person Info)")
    print("=" * 60)
    
    # Define the schema using simple dict format
    person_schema = {
        "type": "OBJECT",
        "properties": {
            "name": {
                "type": "STRING",
                "description": "Person's full name"
            },
            "age": {
                "type": "INTEGER",
                "description": "Person's age"
            },
            "occupation": {
                "type": "STRING",
                "description": "Person's job or occupation"
            }
        },
        "required": ["name", "age", "occupation"]
    }
    
    # Create client with Google provider
    client = LLMClient(
        api_key=os.environ.get("GOOGLE_API_KEY", "your-api-key"),
        base_url="https://generativelanguage.googleapis.com",
        model="gemini-2.0-flash",
        provider="google"
    )
    
    messages = [
        {"role": "user", "content": "Tell me about Albert Einstein in a structured format."}
    ]
    
    # Call chat with schema
    response = client.chat(messages, schema=person_schema)
    print(f"\nRaw response:\n{response}")
    
    # Parse JSON response
    data = json.loads(response)
    print(f"\nParsed data:")
    print(f"  Name: {data['name']}")
    print(f"  Age: {data['age']}")
    print(f"  Occupation: {data['occupation']}")


def demo_array_structured_output():
    """Structured output with array - listing multiple items."""
    print("\n" + "=" * 60)
    print("Demo 2: Array Structured Output (List of Books)")
    print("=" * 60)
    
    # Define schema for a list of books
    books_schema = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "title": {
                    "type": "STRING",
                    "description": "Book title"
                },
                "author": {
                    "type": "STRING",
                    "description": "Author name"
                },
                "year": {
                    "type": "INTEGER",
                    "description": "Publication year"
                },
                "genre": {
                    "type": "STRING",
                    "description": "Book genre"
                }
            },
            "required": ["title", "author", "year"]
        }
    }
    
    client = LLMClient(
        api_key=os.environ.get("GOOGLE_API_KEY", "your-api-key"),
        base_url="https://generativelanguage.googleapis.com",
        model="gemini-2.0-flash",
        provider="google"
    )
    
    messages = [
        {"role": "user", "content": "List 3 famous science fiction books."}
    ]
    
    response = client.chat(messages, schema=books_schema)
    print(f"\nRaw response:\n{response}")
    
    books = json.loads(response)
    print(f"\nParsed books:")
    for i, book in enumerate(books, 1):
        print(f"  {i}. {book['title']} by {book['author']} ({book['year']})")


def demo_nested_structured_output():
    """Structured output with nested objects."""
    print("\n" + "=" * 60)
    print("Demo 3: Nested Structured Output (Company Info)")
    print("=" * 60)
    
    # Define schema with nested structure
    company_schema = {
        "type": "OBJECT",
        "properties": {
            "company_name": {"type": "STRING"},
            "founded_year": {"type": "INTEGER"},
            "headquarters": {
                "type": "OBJECT",
                "properties": {
                    "city": {"type": "STRING"},
                    "country": {"type": "STRING"}
                },
                "required": ["city", "country"]
            },
            "products": {
                "type": "ARRAY",
                "items": {"type": "STRING"}
            },
            "ceo": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "since_year": {"type": "INTEGER"}
                }
            }
        },
        "required": ["company_name", "founded_year", "headquarters"]
    }
    
    client = LLMClient(
        api_key=os.environ.get("GOOGLE_API_KEY", "your-api-key"),
        base_url="https://generativelanguage.googleapis.com",
        model="gemini-2.0-flash",
        provider="google"
    )
    
    messages = [
        {"role": "user", "content": "Give me structured information about Apple Inc."}
    ]
    
    response = client.chat(messages, schema=company_schema)
    print(f"\nRaw response:\n{response}")
    
    company = json.loads(response)
    print(f"\nParsed company info:")
    print(f"  Company: {company['company_name']}")
    print(f"  Founded: {company['founded_year']}")
    print(f"  HQ: {company['headquarters']['city']}, {company['headquarters']['country']}")
    if 'products' in company:
        print(f"  Products: {', '.join(company['products'])}")
    if 'ceo' in company:
        print(f"  CEO: {company['ceo'].get('name', 'N/A')}")


def demo_stream_structured_output():
    """Streaming structured output example."""
    print("\n" + "=" * 60)
    print("Demo 4: Streaming Structured Output")
    print("=" * 60)
    
    # Define simple schema
    recipe_schema = {
        "type": "OBJECT",
        "properties": {
            "dish_name": {"type": "STRING"},
            "cuisine": {"type": "STRING"},
            "prep_time_minutes": {"type": "INTEGER"},
            "ingredients": {
                "type": "ARRAY",
                "items": {"type": "STRING"}
            },
            "steps": {
                "type": "ARRAY",
                "items": {"type": "STRING"}
            }
        },
        "required": ["dish_name", "ingredients", "steps"]
    }
    
    client = LLMClient(
        api_key=os.environ.get("GOOGLE_API_KEY", "your-api-key"),
        base_url="https://generativelanguage.googleapis.com",
        model="gemini-2.0-flash",
        provider="google"
    )
    
    messages = [
        {"role": "user", "content": "Give me a simple pasta recipe."}
    ]
    
    print("\nStreaming response:")
    full_response = ""
    for chunk in client.stream_chat(messages, schema=recipe_schema):
        print(chunk, end="", flush=True)
        full_response += chunk
    
    print("\n\nParsed recipe:")
    recipe = json.loads(full_response)
    print(f"  Dish: {recipe['dish_name']}")
    if 'cuisine' in recipe:
        print(f"  Cuisine: {recipe['cuisine']}")
    if 'prep_time_minutes' in recipe:
        print(f"  Prep time: {recipe['prep_time_minutes']} minutes")
    print(f"  Ingredients: {len(recipe['ingredients'])} items")
    print(f"  Steps: {len(recipe['steps'])} steps")


def demo_with_conversation():
    """Using structured output with Conversation class."""
    print("\n" + "=" * 60)
    print("Demo 5: Structured Output with Conversation")
    print("=" * 60)
    
    from conversation import Conversation
    
    sentiment_schema = {
        "type": "OBJECT",
        "properties": {
            "text": {
                "type": "STRING",
                "description": "The analyzed text"
            },
            "sentiment": {
                "type": "STRING",
                "description": "Overall sentiment: positive, negative, neutral, or mixed"
            },
            "confidence": {
                "type": "NUMBER",
                "description": "Confidence score 0-1"
            },
            "key_phrases": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
                "description": "Key phrases that influenced the sentiment"
            }
        },
        "required": ["text", "sentiment", "confidence"]
    }
    
    client = LLMClient(
        api_key=os.environ.get("GOOGLE_API_KEY", "your-api-key"),
        base_url="https://generativelanguage.googleapis.com",
        model="gemini-2.0-flash",
        provider="google"
    )
    
    conversation = Conversation(
        client, 
        system_prompt="You are a sentiment analysis assistant. Analyze the sentiment of user messages."
    )
    
    # Using conversation.send with schema passed through kwargs
    response = conversation.send(
        "I absolutely loved the new restaurant! The food was amazing but the service was a bit slow.",
        schema=sentiment_schema
    )
    
    print(f"\nRaw response:\n{response}")
    
    result = json.loads(response)
    print(f"\nSentiment Analysis:")
    print(f"  Sentiment: {result['sentiment']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    if 'key_phrases' in result:
        print(f"  Key phrases: {', '.join(result['key_phrases'])}")


def demo_from_yaml_file():
    """Demo loading schema from YAML file."""
    print("\n" + "=" * 60)
    print("Demo 6: Load Schema from YAML File")
    print("=" * 60)
    
    # Example YAML content (normally you would load this from a file)
    yaml_content = """
schema:
  type: "OBJECT"
  properties:
    reasoning:
      type: "STRING"
      description: "对当前环境的分析和动作规划的逻辑说明。"
    action:
      type: "ARRAY"
      description: "一系列原子化动作指令列表。"
      items:
        type: "STRING"
        description: "简洁的动作描述，如 'Turn right', 'Go straight' 等。"
  required:
    - "reasoning"
    - "action"
"""
    
    import yaml
    config = yaml.safe_load(yaml_content)
    schema = config["schema"]
    
    print(f"Loaded schema from YAML:")
    print(json.dumps(schema, indent=2, ensure_ascii=False))
    
    client = LLMClient(
        api_key=os.environ.get("GOOGLE_API_KEY", "your-api-key"),
        base_url="https://generativelanguage.googleapis.com",
        model="gemini-2.0-flash",
        provider="google"
    )
    
    messages = [
        {"role": "user", "content": "你看到前方有一个红色的球，请规划如何接近它。"}
    ]
    
    response = client.chat(messages, schema=schema)
    print(f"\nResponse:\n{response}")
    
    data = json.loads(response)
    print(f"\nParsed result:")
    print(f"  Reasoning: {data['reasoning']}")
    print(f"  Actions: {data['action']}")


if __name__ == "__main__":
    # Check for API key
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Warning: GOOGLE_API_KEY environment variable not set.")
        print("Set it with: export GOOGLE_API_KEY=your-api-key")
        print("Or on Windows: set GOOGLE_API_KEY=your-api-key")
        print("\nRunning demos anyway (will fail without valid API key)...\n")
    
    try:
        demo_basic_structured_output()
        demo_array_structured_output()
        demo_nested_structured_output()
        demo_stream_structured_output()
        demo_with_conversation()
        demo_from_yaml_file()
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure you have set a valid GOOGLE_API_KEY environment variable.")
