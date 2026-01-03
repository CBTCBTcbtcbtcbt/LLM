from google.genai import types
from PIL import Image

try:
    img = Image.new('RGB', (100, 100), color='red')
    
    # Test 1: Can we put image in a list for Part? No, Part expects specific args usually.
    # But maybe types.Part(img) works?
    try:
        part = types.Part(img)
        print("types.Part(img) worked")
    except Exception as e:
        print(f"types.Part(img) failed: {e}")

    # Test 2: Can we put image in parts list for Content?
    try:
        content = types.Content(parts=["text", img])
        print("types.Content(parts=['text', img]) worked")
    except Exception as e:
        print(f"types.Content(parts=['text', img]) failed: {e}")

    # Test 3: Maybe types.Part.from_image?
    # I don't know if it exists, check dir
    print("types.Part attributes:", dir(types.Part))

except Exception as e:
    print(f"General error: {e}")
