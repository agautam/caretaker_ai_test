"""
AI Technician - Function Calling Agent using Gemini to analyze HVAC data.
"""

import os
import json
import requests
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool
from dotenv import load_dotenv
import chromadb

# Load environment variables from .env file
load_dotenv()

# Debug mode flag
DEBUG_MODE = True

def get_sensor_data():
    """
    Fetches sensor data from the HVAC API diagnose endpoint.
    
    Returns:
    --------
    dict
        JSON response from the diagnose endpoint
    """
    if DEBUG_MODE:
        print("[DEBUG] Agent is calling get_sensor_data()...")
    
    base_url = "http://127.0.0.1:8000"
    
    try:
        response = requests.get(f"{base_url}/diagnose")
        response.raise_for_status()
        data = response.json()
        
        if DEBUG_MODE:
            print(f"[DEBUG] get_sensor_data() returned: {json.dumps(data, indent=2)}")
        
        return data
    
    except requests.exceptions.ConnectionError:
        error_msg = {
            "error": "Could not connect to the API. Make sure the FastAPI server is running.",
            "suggestion": "Start it with: uvicorn api:app --reload"
        }
        if DEBUG_MODE:
            print(f"[DEBUG] get_sensor_data() error: {error_msg}")
        return error_msg
    except requests.exceptions.HTTPError as e:
        error_msg = {"error": f"Error fetching data from API: {e}"}
        if DEBUG_MODE:
            print(f"[DEBUG] get_sensor_data() error: {error_msg}")
        return error_msg
    except Exception as e:
        error_msg = {"error": f"Unexpected error: {e}"}
        if DEBUG_MODE:
            print(f"[DEBUG] get_sensor_data() error: {error_msg}")
        return error_msg

def consult_manual(topic: str):
    """
    Queries the ChromaDB manual collection for information about a specific topic.
    
    Parameters:
    -----------
    topic : str
        The topic or error code to search for in the manual
    
    Returns:
    --------
    dict
        Dictionary containing the retrieved text chunks
    """
    if DEBUG_MODE:
        print(f"[DEBUG] Agent is calling consult_manual(topic='{topic}')...")
    
    try:
        # Create a persistent ChromaDB client
        client = chromadb.PersistentClient(path="./chroma_db")
        
        # Get the collection
        collection = client.get_collection(name="hvac_manual")
        
        # Query the collection
        results = collection.query(
            query_texts=[topic],
            n_results=2
        )
        
        # Extract documents from results
        chunks = []
        if results and 'documents' in results and len(results['documents']) > 0:
            chunks = results['documents'][0]
        
        result = {
            "topic": topic,
            "chunks_found": len(chunks),
            "chunks": chunks
        }
        
        if DEBUG_MODE:
            print(f"[DEBUG] consult_manual() returned {len(chunks)} chunks for topic '{topic}'")
            for i, chunk in enumerate(chunks, 1):
                print(f"[DEBUG]   Chunk {i} (first 100 chars): {chunk[:100]}...")
        
        return result
    
    except Exception as e:
        error_msg = {
            "topic": topic,
            "error": f"Could not query ChromaDB collection: {e}",
            "chunks_found": 0,
            "chunks": []
        }
        if DEBUG_MODE:
            print(f"[DEBUG] consult_manual() error: {error_msg}")
        return error_msg

def create_tools():
    """
    Create function declarations and tools for the Gemini model.
    
    Returns:
    --------
    list
        List of Tool objects
    """
    # Define get_sensor_data function
    get_sensor_data_func = FunctionDeclaration(
        name="get_sensor_data",
        description="Fetches sensor data and diagnosis information from the HVAC API diagnose endpoint. Returns JSON data about system status, errors, and short-cycling events.",
        parameters={
            "type": "object",
            "properties": {},
        }
    )
    
    # Define consult_manual function
    consult_manual_func = FunctionDeclaration(
        name="consult_manual",
        description="Queries the HVAC technical manual database for information about a specific error code, symptom, or topic. Use this to find root causes and recommended fixes from the official manual.",
        parameters={
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The error code (e.g., 'Error 001', 'Error 002', 'Error 003'), symptom (e.g., 'Short Cycling', 'Filter Clog', 'Low Refrigerant'), or topic to search for in the manual."
                }
            },
            "required": ["topic"]
        }
    )
    
    # Create tools
    tools = Tool(
        function_declarations=[get_sensor_data_func, consult_manual_func]
    )
    
    return [tools]

def main():
    """Main function to run the AI technician function calling agent."""
    
    print("=" * 70)
    print("AI HVAC Technician - Function Calling Agent")
    print("=" * 70)
    print()
    
    # Get API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment variables.")
        print("Please add it to your .env file.")
        return
    
    # Configure the Gemini API
    genai.configure(api_key=api_key)
    
    # Find available Gemini model
    print("Finding available Gemini model...")
    available_models = genai.list_models()
    
    gemini_models = []
    for model in available_models:
        model_display_name = model.name.replace('models/', '').lower()
        if model_display_name.startswith('gemini'):
            methods = model.supported_generation_methods if hasattr(model, 'supported_generation_methods') and model.supported_generation_methods else []
            if methods and 'generateContent' in methods:
                gemini_models.append(model)
    
    if not gemini_models:
        print("Error: No Gemini models found that support generateContent.")
        return
    
    # Preferred model names in order of preference
    preferred_names = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro', 'gemini-1.5-flash-latest']
    
    model_name = None
    for preferred in preferred_names:
        for model in gemini_models:
            model_display_name = model.name.replace('models/', '').lower()
            if model_display_name == preferred:
                model_name = model.name.replace('models/', '')
                print(f"Using preferred model: {model_name}")
                break
        if model_name:
            break
    
    if not model_name:
        model_name = gemini_models[0].name.replace('models/', '')
        print(f"Using available Gemini model: {model_name}")
    
    # Create tools
    tools = create_tools()
    
    # Initialize the model with tools
    model = genai.GenerativeModel(
        model_name=model_name,
        tools=tools,
        system_instruction=(
            "You are a Senior HVAC Forensic Analyst. Your goal is to explain failure modes, not just fix them.\n\n"
            "Step 1: Check sensor data.\n"
            "Step 2: If errors exist, consult the manual for the SPECIFIC error or symptom.\n"
            "Step 3: Draft a response that explicitly lists: a) The Observed Symptom. b) The ROOT CAUSE (found in the manual). c) The Recommended Fix.\n\n"
            "Do not assume root causes; look them up."
        )
    )
    
    # Start a chat session
    chat = model.start_chat()
    
    # Function call handler
    def handle_function_call(function_name, args):
        """Handle function calls from the model."""
        if function_name == "get_sensor_data":
            return get_sensor_data()
        elif function_name == "consult_manual":
            topic = args.get("topic", "")
            return consult_manual(topic)
        else:
            return {"error": f"Unknown function: {function_name}"}
    
    # Send the user message
    user_message = "System diagnosis requested."
    print(f"\nUser message: {user_message}")
    print("\n" + "=" * 70)
    print("AGENT EXECUTION (Debug Mode)")
    print("=" * 70)
    print()
    
    # Send message and handle function calls in a loop
    response = chat.send_message(user_message)
    max_iterations = 10  # Prevent infinite loops
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        # Check if response contains function calls
        if not response.candidates:
            break
        
        candidate = response.candidates[0]
        if not hasattr(candidate, 'content') or not candidate.content:
            break
        
        # Check if any part is a function call
        function_call_found = False
        for part in candidate.content.parts:
            if hasattr(part, 'function_call') and part.function_call:
                function_call_found = True
                function_call = part.function_call
                function_name = function_call.name
                args = {}
                
                # Extract arguments - handle different possible structures
                if hasattr(function_call, 'args'):
                    if function_call.args:
                        # Convert to dict if it's a protobuf message
                        try:
                            args = dict(function_call.args)
                        except (TypeError, AttributeError):
                            # If it's already a dict or different structure
                            args = function_call.args if isinstance(function_call.args, dict) else {}
                
                # Call the function
                function_result = handle_function_call(function_name, args)
                
                # Create function response part
                try:
                    from google.generativeai.types import FunctionResponse
                    function_response_part = FunctionResponse(
                        name=function_name,
                        response=function_result
                    )
                except ImportError:
                    # Fallback to protos if FunctionResponse not available
                    function_response_part = genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=function_name,
                            response=function_result
                        )
                    )
                
                # Send function result back to the model
                response = chat.send_message(function_response_part)
                break
        
        if not function_call_found:
            break
    
    # Print the final response
    print("\n" + "=" * 70)
    print("FINAL DIAGNOSIS")
    print("=" * 70)
    print()
    
    # Extract text from response safely
    try:
        response_text = response.text
    except AttributeError:
        # Fallback: try to extract text from parts
        response_text = ""
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text') and part.text:
                    response_text += part.text
    
    print(response_text)
    print("=" * 70)

if __name__ == '__main__':
    main()
