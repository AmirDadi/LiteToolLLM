"""
Simple example demonstrating LiteCallLLM usage.

This example shows how to use the structured_completion function with a Pydantic model
to validate the output structure and use tool calling.
"""

import os
import sys
from typing import List
from pydantic import BaseModel
from litecallllm import structured_completion, Tool

# Define a Pydantic model for the structured output
class Weather(BaseModel):
    location: str
    temperature: str
    description: str

# Define a tool function
def get_weather(location: str) -> dict:
    """Get the current weather in a location.
    
    Args:
        location: The city and state, e.g. San Francisco, CA
        
    Returns:
        A dictionary with weather information
    """
    # In a real application, this would call a weather API
    weather_data = {
        "San Francisco": {"temperature": "68°F", "description": "Foggy"},
        "New York": {"temperature": "72°F", "description": "Sunny"},
        "Chicago": {"temperature": "65°F", "description": "Windy"},
    }
    
    # Return data for the requested location or a default response
    for city in weather_data:
        if city.lower() in location.lower():
            return {
                "location": city,
                "temperature": weather_data[city]["temperature"],
                "description": weather_data[city]["description"]
            }
    
    # Default response for unknown locations
    return {
        "location": location,
        "temperature": "Unknown",
        "description": "No data available"
    }

def main():
    # Example using structured_completion with a simple query
    try:
        # Set your API key (assuming OpenAI for this example)
        os.environ["OPENAI_API_KEY"] = "your-api-key-here"
        
        # Create a message asking about weather
        messages = [
            {"role": "user", "content": "What's the weather like in San Francisco and Chicago?"}
        ]
        
        # Use structured_completion with the Weather model and get_weather tool
        response = structured_completion(
            model="gpt-4o",  # Supported model with function calling capabilities
            messages=messages,
            response_model=Weather,
            tools=[get_weather],
            max_recursion=5
        )
        
        # Print the structured response
        print("\nStructured Response:")
        print(f"Content: {response.content}")
        print("\nFull conversation history:")
        for msg in response.messages:
            print(f"[{msg.get('role')}]: {msg.get('content')}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 