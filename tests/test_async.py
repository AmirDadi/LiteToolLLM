import os
import pytest
from litetoolllm.core import astructured_completion
from litetoolllm.models import Temperature, Temperatures
from litetoolllm.tools import aget_current_weather, aconvert_fahrenheit_to_celsius

pytestmark = pytest.mark.asyncio

class TestAsyncFunctionCalling:
    async def test_single_tool_execution(self):
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No API key available")

        response = await astructured_completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is the weather in San Francisco?"}],
            response_model=Temperature,
            tools=[aget_current_weather],
            max_recursion=10
        )
        assert len(response.messages) == 4
        assert isinstance(response.content, Temperature)

    async def test_single_tool_execution_without_schema(self):
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No API key available")

        response = await astructured_completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is the weather in San Francisco?"}],
            tools=[aget_current_weather],
            max_recursion=10
        )
        assert len(response.messages) == 4
        assert isinstance(response.content, str)

    async def test_multiple_tool_parallel(self):
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No API key available")

        response = await astructured_completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is the weather in San Francisco and New york?"}],
            response_model=Temperatures,
            tools=[aget_current_weather],
            max_recursion=10,
            parallel_tool_calls=True
        )
        assert len(response.messages) == 5
        assert isinstance(response.content, Temperatures)

    async def test_multiple_tool_sequential(self):
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No API key available")

        response = await astructured_completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is the weather in San Francisco in celsius?"}],
            response_model=Temperature,
            tools=[aget_current_weather, aconvert_fahrenheit_to_celsius],
            max_recursion=10
        )
        assert len(response.messages) == 6
        assert isinstance(response.content, Temperature)

class TestAsyncStructuredOutput:
    async def test_basic_structured_response(self):
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No API key available")

        response = await astructured_completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Tell me the temperature."}],
            response_model=Temperature,
            tools=None,
            mock_response='{"temperature": "68", "location": "None"}'
        )
        assert len(response.messages) == 2
        assert isinstance(response.content, Temperature) 