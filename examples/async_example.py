"""
Async example demonstrating liteToolLlm usage.

This example shows how to use the astructured_completion function with
asynchronous tool functions and parallel tool execution.
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from litetoolllm import astructured_completion, Tool
# Define a Pydantic model for multiple weather locations
class WeatherReport(BaseModel):
    locations: Optional[List[Dict[str, str]]]
    summary: Optional[str]

# Define an async tool function
async def async_get_weather(location: str) -> dict:
    """Get the current weather in a location asynchronously.
    
    Args:
        location: The city and state, e.g. San Francisco, CA
        
    Returns:
        A dictionary with weather information
    """
    # Simulate API call with delay
    await asyncio.sleep(1)
    
    # In a real application, this would call a weather API
    weather_data = {
        "San Francisco": {"temperature": "68°F", "description": "Foggy"},
        "New York": {"temperature": "72°F", "description": "Sunny"},
        "Chicago": {"temperature": "65°F", "description": "Windy"},
        "Los Angeles": {"temperature": "75°F", "description": "Clear"},
        "Miami": {"temperature": "85°F", "description": "Humid"},
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

async def main():
    # Example using astructured_completion with parallel tool calling
    try:
        # Set your API key (assuming OpenAI for this example)
        os.environ["OPENAI_API_KEY"] = "your-api-key-here"
        
        # Create a message asking about weather in multiple cities
        messages = [
            {"role": "user", "content": "What's the weather like in San Francisco, Chicago, Miami, and New York? Give me a summary."}
        ]
        
        # Create a Tool instance for better control
        weather_tool = Tool(
            func=async_get_weather,
            name="get_weather",
            description="Get weather information for a specific location"
        )
        
        # Use astructured_completion with parallel tool execution
        print("Starting weather query (this will take a moment)...")
        response = await astructured_completion(
            model="gpt-4o-mini",  # Supported model with function calling capabilities
            messages=messages,
            response_model=WeatherReport,
            tools=[weather_tool],
            max_recursion=5,
            parallel_tool_calls=True  # Enable parallel execution
        )
        
        # Print the structured response
        print("\nStructured Response:")
        print(f"Locations: {response.content.locations}")
        print(f"Summary: {response.content.summary}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 