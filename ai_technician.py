"""
AI Technician - Uses Gemini Flash to analyze HVAC data and provide recommendations.
"""

import os
import requests
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def fetch_hvac_data():
    """
    Fetch data from the HVAC API endpoints.
    
    Returns:
    --------
    tuple: (latest_data, diagnose_data)
    """
    base_url = "http://127.0.0.1:8000"
    
    try:
        # Fetch latest sensor reading
        print("Fetching latest sensor data...")
        latest_response = requests.get(f"{base_url}/latest")
        latest_response.raise_for_status()
        latest_data = latest_response.json()
        
        # Fetch short-cycling diagnosis
        print("Fetching short-cycling diagnosis...")
        diagnose_response = requests.get(f"{base_url}/diagnose")
        diagnose_response.raise_for_status()
        diagnose_data = diagnose_response.json()
        
        return latest_data, diagnose_data
    
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the FastAPI server is running.")
        print("Start it with: uvicorn api:app --reload")
        return None, None
    except requests.exceptions.HTTPError as e:
        print(f"Error fetching data from API: {e}")
        return None, None

def format_data_for_ai(latest_data, diagnose_data):
    """
    Format the HVAC data into a readable string for the AI.
    
    Parameters:
    -----------
    latest_data : dict
        Latest sensor reading
    diagnose_data : dict
        Short-cycling diagnosis results
    
    Returns:
    --------
    str
        Formatted data string
    """
    data_str = "HVAC Sensor Data Analysis\n"
    data_str += "=" * 50 + "\n\n"
    
    # Latest sensor reading
    data_str += "LATEST SENSOR READING:\n"
    data_str += f"  Timestamp: {latest_data.get('timestamp', 'N/A')}\n"
    data_str += f"  Unit ID: {latest_data.get('unit_id', 'N/A')}\n"
    data_str += f"  Temperature: {latest_data.get('temperature', 'N/A')}°F\n"
    data_str += f"  Humidity: {latest_data.get('humidity', 'N/A')}%\n"
    data_str += f"  Energy Consumption: {latest_data.get('energy_consumption', 'N/A')} kW\n"
    data_str += f"  Error Code: {latest_data.get('error_code', 'N/A')}\n\n"
    
    # Short-cycling diagnosis
    data_str += "SHORT-CYCLING DIAGNOSIS:\n"
    if not diagnose_data or len(diagnose_data) == 0:
        data_str += "  No short-cycling events detected.\n"
    else:
        data_str += f"  Short-cycling detected in {len(diagnose_data)} unit(s):\n\n"
        for unit_id, timestamps in diagnose_data.items():
            data_str += f"  Unit: {unit_id}\n"
            data_str += f"  Number of short-cycling transitions: {len(timestamps)}\n"
            data_str += f"  Timestamps:\n"
            for ts in timestamps[:10]:  # Show first 10 timestamps
                data_str += f"    - {ts}\n"
            if len(timestamps) > 10:
                data_str += f"    ... and {len(timestamps) - 10} more\n"
            data_str += "\n"
    
    return data_str

def analyze_with_gemini(data_str):
    """
    Send data to Gemini Flash model for analysis.
    
    Parameters:
    -----------
    data_str : str
        Formatted HVAC data string
    
    Returns:
    --------
    str
        AI response
    """
    # Get API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found in environment variables. "
            "Please add it to your .env file."
        )
    
    # Configure the Gemini API
    genai.configure(api_key=api_key)
    
    # List available models and find one that supports generateContent
    print("\nFinding available Gemini model...")
    available_models = genai.list_models()
    
    # Filter models to only those starting with "gemini" (case insensitive) and print them
    print("\nAvailable Gemini models:")
    print("-" * 70)
    gemini_models = []
    for model in available_models:
        model_display_name = model.name.replace('models/', '').lower()
        if model_display_name.startswith('gemini'):
            # Check if generateContent is supported
            methods = model.supported_generation_methods if hasattr(model, 'supported_generation_methods') and model.supported_generation_methods else []
            methods_str = ', '.join(methods) if methods else 'None'
            print(f"  {model.name.replace('models/', '')}")
            print(f"    Supported methods: {methods_str}")
            if methods and 'generateContent' in methods:
                gemini_models.append(model)
    print("-" * 70)
    print()
    
    if not gemini_models:
        raise ValueError(
            "No Gemini models found that support generateContent. "
            "Please check your API key and available models."
        )
    
    # Preferred model names in order of preference
    preferred_names = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro', 'gemini-1.5-flash-latest']
    
    model_name = None
    for preferred in preferred_names:
        for model in gemini_models:
            # Check if model name matches (with or without 'models/' prefix)
            model_display_name = model.name.replace('models/', '').lower()
            if model_display_name == preferred:
                model_name = model.name.replace('models/', '')
                print(f"Using preferred model: {model_name}")
                break
        if model_name:
            break
    
    # If no preferred model found, use the first available Gemini model that supports generateContent
    if not model_name:
        model_name = gemini_models[0].name.replace('models/', '')
        print(f"Using available Gemini model: {model_name}")
    
    if not model_name:
        raise ValueError(
            "No available Gemini model found that supports generateContent. "
            "Please check your API key and available models."
        )
    
    # Initialize the model
    model = genai.GenerativeModel(model_name)
    
    # System prompt
    system_prompt = (
        "You are a senior HVAC technician. Analyze the data. "
        "If short-cycling is detected, explain WHY it is bad (wear and tear) "
        "and recommend a fix to the homeowner. Keep it professional but urgent."
    )
    
    # Combine system prompt with data
    full_prompt = f"{system_prompt}\n\n{data_str}"
    
    print("Sending data to Gemini for analysis...")
    
    # Generate response
    response = model.generate_content(full_prompt)
    
    return response.text

def main():
    """Main function to run the AI technician analysis."""
    
    print("=" * 70)
    print("AI HVAC Technician - Analysis")
    print("=" * 70)
    print()
    
    # Fetch data from API
    latest_data, diagnose_data = fetch_hvac_data()
    
    if latest_data is None or diagnose_data is None:
        return
    
    # Format data for AI
    data_str = format_data_for_ai(latest_data, diagnose_data)
    
    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)
    print(data_str)
    
    # Analyze with Gemini
    try:
        ai_response = analyze_with_gemini(data_str)
        
        print("\n" + "=" * 70)
        print("AI TECHNICIAN ANALYSIS")
        print("=" * 70)
        print(ai_response)
        print("=" * 70)
        
    except ValueError as e:
        print(f"\nError: {e}")
        print("\nTo fix this:")
        print("1. Create a .env file in the project root")
        print("2. Add your Google API key: GEMINI_API_KEY=your_api_key_here")
    except Exception as e:
        print(f"\nError analyzing with Gemini: {e}")

if __name__ == '__main__':
    main()

