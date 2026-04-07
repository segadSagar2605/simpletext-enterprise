import os
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("CRITICAL: GEMINI_API_KEY not found in .env file.")
else:
    client = genai.Client(api_key=api_key)
    print("--- ANALYZING YOUR AVAILABLE MODELS ---")
    
    try:
        # Using the new SDK's attribute: supported_actions
        for m in client.models.list():
            if 'embedContent' in m.supported_actions:
                print(f"FOUND: {m.name}")
    except Exception as e:
        print(f"Error: {e}")
        print("\nFallback: Listing ALL available models for you:")
        # If the filter fails, just list everything so we can see the names
        for m in client.models.list():
            print(f" - {m.name}")