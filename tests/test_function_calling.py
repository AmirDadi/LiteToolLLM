# tests/test_function_calling.py
import os
import pytest
import litellm

from litetoolllm.errors import ModelCapabilityError, StructuredValidationError
from litetoolllm.core import structured_completion
from litetoolllm.models import Temperature, Temperatures
from litetoolllm.tools import get_current_weather, convert_fahrenheit_to_celsius
class TestFunctionCalling:
    def test_single_tool_execution(self):
        """Test basic tool calling with weather lookup"""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No API key available")

        response = structured_completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is the weather in San Francisco?"}],
            response_model=Temperature,
            tools=[get_current_weather],
            max_recursion=10
        )
        assert len(response.messages) == 4
        assert isinstance(response.content, Temperature)

    def test_single_tool_execution_without_schema(self):
        """Test basic tool calling with weather lookup without schema"""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No API key available")

        response = structured_completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is the weather in San Francisco?"}],
            tools=[get_current_weather],
            max_recursion=10
        )
        assert len(response.messages) == 4
        assert isinstance(response.content, str)

    def test_multiple_tool_parallel(self):
        """Test tool calling with weather lookup with parallel tool calling"""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No API key available")

        response = structured_completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is the weather in San Francisco and New york?"}],
            response_model=Temperatures,
            tools=[get_current_weather],
            max_recursion=10,
            parallel_tool_calls=True
        )
        assert len(response.messages) == 5
        assert isinstance(response.content, Temperatures)

    def test_multiple_tool_sequential(self):
        """Test chained tool calls (Fahrenheit to Celsius conversion)"""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No API key available")

        response = structured_completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is the weather in San Francisco in celsius?"}],
            response_model=Temperature,
            tools=[get_current_weather, convert_fahrenheit_to_celsius],
            max_recursion=10
        )
        assert len(response.messages) == 6
        assert isinstance(response.content, Temperature)

class TestStructuredOutput:
    def test_basic_structured_response(self):
        """Test structured output without tool usage"""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No API key available")

        response = structured_completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Tell me the temperature."}],
            response_model=Temperature,
            tools=None,
            mock_response='{"temperature": "68", "location": "None"}'
        )
        assert len(response.messages) == 2
        assert isinstance(response.content, Temperature)

class TestErrorHandling:
    def test_model_without_json_support_raises_error(self, monkeypatch):
        """Verify error when using response_model with non-json model"""
        monkeypatch.setattr(litellm, "supports_response_schema", lambda model: False)

        with pytest.raises(ModelCapabilityError):
            structured_completion(
                model="non-json-model",
                messages=[{"role": "user", "content": "Tell me something."}],
                response_model=Temperature,
                tools=None
            )

    def test_model_without_tool_support_raises_error(self, monkeypatch):
        """Verify error when using tools with non-tool-enabled model"""
        monkeypatch.setattr(litellm, "supports_function_calling", lambda model: False)

        with pytest.raises(ModelCapabilityError):
            structured_completion(
                model="non-tool-model",
                messages=[{"role": "user", "content": "Execute tool call."}],
                response_model=Temperature,
                tools=[get_current_weather]
            )

class TestValidation:
    def test_invalid_response_structure(self, monkeypatch):
        """Test handling of malformed model responses"""
        def mock_invalid_completion(*args, **kwargs):
            return {"choices": [{"message": {"content": 'invalid json'}}]}

        monkeypatch.setattr("litetoolllm.core.completion", mock_invalid_completion)

        with pytest.raises(StructuredValidationError):
            structured_completion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Provide temperature."}],
                response_model=Temperature,
                tools=None
            )
