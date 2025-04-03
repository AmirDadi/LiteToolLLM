# structurallm/tools.py
def get_current_weather(location: str) -> dict:
    """
    Return weather
    :param location: str
    :return: dict
    """
    if "San Francisco" in location:
        return {"location": "San Francisco", "temperature": "68°F"}
    return {"location": location, "temperature": "unknown"}


def convert_fahrenheit_to_celsius(temp_in_fahrenheit: str) -> dict:
    """
    Return weather
    :param temp_in_fahrenheit: str
    :return: str
    """
    return "30 Cel"