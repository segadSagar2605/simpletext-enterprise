import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("CRITICAL: GEMINI_API_KEY not found in .env file.")
else:
    client = genai.Client(api_key=api_key)
    
    print("\n--- EMBEDDING MODELS ---")
    for m in client.models.list():
        if 'embedContent' in m.supported_actions:
            print(f"  {m.name}")

    print("\n--- GENERATION MODELS ---")
    for m in client.models.list():
        if 'generateContent' in m.supported_actions:
            print(f"  {m.name}")

    print("\n--- TESTING gemini-2.5-flash-lite ---")
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents="Say hello in one word."
        )
        print(f"  SUCCESS: {response.text.strip()}")
    except Exception as e:
        print(f"  FAILED: {e}")