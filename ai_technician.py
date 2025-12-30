import os
import json
import requests
import time
from dotenv import load_dotenv
import chromadb
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# NEW SDK IMPORTS
from google import genai
from google.genai import types
from google.genai import errors

# Load environment variables from .env file
load_dotenv()

# Debug mode flag
DEBUG_MODE = True

# --- GLOBAL HELPER FUNCTIONS (The "Hands") ---

def get_digital_twin_status():
    """
    Fetches the Digital Twin health object from the Hubbub API.
    """
    unit_id = "UNIT_001" 
    base_url = "http://127.0.0.1:8000"
    
    if DEBUG_MODE:
        print(f"[DEBUG] Agent is fetching Digital Twin for {unit_id}...")
    
    try:
        response = requests.get(f"{base_url}/health/{unit_id}")
        response.raise_for_status()
        data = response.json()
        if DEBUG_MODE:
            print(f"[DEBUG] Twin Status: {data.get('status')} | Anomalies: {len(data.get('anomalies', []))}")
        return data
    except Exception as e:
        return {"error": f"Connection failed: {e}", "status": "UNKNOWN"}

def consult_manual(topic: str):
    """
    Queries the ChromaDB manual collection for information about a specific topic.
    """
    if DEBUG_MODE:
        print(f"[DEBUG] Agent is consulting manual for: '{topic}'...")
    
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection(name="hvac_manual")
        results = collection.query(query_texts=[topic], n_results=2)
        chunks = results['documents'][0] if results['documents'] else []
        return {"topic": topic, "manual_extracts": chunks}
    except Exception as e:
        return {"error": f"Manual lookup failed: {e}"}

# --- TOOL DEFINITIONS ---
tools_list = [get_digital_twin_status, consult_manual]

# --- MAIN AGENT LOGIC WITH RETRY ---

# Retry configuration: 
# Wait 2^x * 1 second between retries (2s, 4s, 8s...)
# Stop after 5 attempts
@retry(
    retry=retry_if_exception_type(errors.ClientError), 
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5)
)
def run_chat_session(client, model_name):
    """Executes the chat session with retry logic for 429 errors."""
    
    system_instruction = (
        "You are a Senior HVAC Forensic Analyst.\n"
        "1. ALWAYS call `get_digital_twin_status` first.\n"
        "2. Check the `status` field.\n"
        "   - If 'HEALTHY': Inform the user the system is normal. Do NOT check the manual.\n"
        "   - If 'CRITICAL' or 'WARNING': Read the `anomalies` list.\n"
        "3. If anomalies exist, call `consult_manual` for the specific `issue_type`.\n"
        "4. Final Report: Combine the Twin data + Manual info to explain the root cause."
    )

    print(f"[System] Attempting to connect to model: {model_name}")
    
    chat = client.chats.create(
        model=model_name, 
        config=types.GenerateContentConfig(
            tools=tools_list,
            system_instruction=system_instruction,
            temperature=0.0,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=False,
                maximum_remote_calls=5
            )
        )
    )

    response = chat.send_message("Diagnose the system.")
    return response

def main():
    print("=" * 70)
    print("AI HVAC Technician - Digital Twin Agent (New SDK + Retry)")
    print("=" * 70)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY missing.")
        return

    client = genai.Client(api_key=api_key)
    
    # Fallback Strategy: Try newer model, if rate limited repeatedly, maybe user should switch manually
    # But for now, let's stick to 2.0 Flash as requested, but with retries.
    # If 2.0 fails consistently, change this string to 'gemini-1.5-flash'
    #model_name = 'gemini-2.0-flash' 
    model_name = 'gemini-2.5-flash'

    print("\n[System] User: 'Diagnose the system.'")
    
    try:
        response = run_chat_session(client, model_name)
        print("\n" + "=" * 70)
        print("FINAL DIAGNOSIS")
        print("=" * 70)
        print(response.text)

    except Exception as e:
        print(f"\n[CRITICAL] Agent failed after retries: {e}")
        print("Tip: If you are seeing 429 Resource Exhausted, wait 60 seconds or switch to 'gemini-1.5-flash'.")

if __name__ == '__main__':
    main()