# litetoolllm/tools.py
from typing import Callable, Dict, Any, Optional

class Tool:
    """
    A class to represent a callable tool for LLM function calling.
    
    Attributes:
        func (Callable): The function to be called
        name (str): The name of the tool (defaults to function name)
        description (str): Description of what the tool does
        parameters (Dict[str, Any]): Parameters schema for the tool
    """
    def __init__(self, 
                 func: Callable, 
                 name: Optional[str] = None, 
                 description: Optional[str] = None,
                 parameters: Optional[Dict[str, Any]] = None):
        self.func = func
        self.name = name or func.__name__
        self.description = description or func.__doc__ or ""
        self.parameters = parameters or {}
    
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

def get_current_weather(location: str, metadata: dict) -> dict:
    """
    Return weather
    :param location: str
    :return: dict
    """
    if "San Francisco" in location:
        return {"location": "San Francisco", "temperature": "68°F"}
    return {"location": location, "temperature": "unknown"}

async def aget_current_weather(location: str, metadata: dict) -> dict:
    """
    Return weather
    :param location: str
    :return: dict
    """
    if "San Francisco" in location:
        return {"location": "San Francisco", "temperature": "68°F"}
    return {"location": location, "temperature": "unknown"}

def convert_fahrenheit_to_celsius(temp_in_fahrenheit: str, metadata: dict) -> dict:
    """
    Return weather
    :param temp_in_fahrenheit: str
    :return: str
    """
    return "30 Cel"

async def aconvert_fahrenheit_to_celsius(temp_in_fahrenheit: str, metadata: dict) -> dict:
    """
    Return weather
    :param temp_in_fahrenheit: str
    :return: str
    """
    return "30 Cel"