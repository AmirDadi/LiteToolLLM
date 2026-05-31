# tests/test_tool_output_live.py
"""Live tests for the result-function (tool-output) path.

These hit real providers through OpenRouter / Gemini and auto-skip unless the
relevant API key is present. They exercise the routing rows that the
deterministic suite can only simulate:

  * TOOL_OUTPUT on a model with tool calling but no JSON-schema support
    (Anthropic, Llama/Groq via OpenRouter).
  * User tools + response_model together (the path that replaces the old
    Gemini two-call hack).
  * Prose fallback -> forced extraction.
  * Gemini tools + response_model through the standard JSON path (proving the
    deleted special-case is covered).
"""
import os

import pytest

from litetoolllm.core import structured_completion, astructured_completion
from litetoolllm.models import Temperature
from litetoolllm.tools import get_current_weather

OPENROUTER = pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not set"
)
GEMINI = pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set"
)


@OPENROUTER
class TestToolOutputLive:
    def test_anthropic_structured_via_tool_output(self):
        """Anthropic (tools yes, JSON schema no) returns structured output."""
        response = structured_completion(
            model="openrouter/anthropic/claude-sonnet-4.6",
            messages=[{"role": "user", "content": "The temperature in Paris is 20C. Report it."}],
            response_model=Temperature,
            max_recursion=5,
        )
        assert isinstance(response.content, Temperature)

    def test_llama_structured_with_capability_override(self):
        """Llama via OpenRouter; litellm misreports function_calling, so override it."""
        response = structured_completion(
            model="openrouter/meta-llama/llama-3.3-70b-instruct",
            messages=[{"role": "user", "content": "The temperature in Tokyo is 25C. Report it."}],
            response_model=Temperature,
            model_capabilities={"function_calling": True, "json_mode": False},
            max_recursion=5,
        )
        assert isinstance(response.content, Temperature)

    def test_user_tools_plus_response_model(self):
        """User tool + structured output together via the tool-output path."""
        response = structured_completion(
            model="openrouter/anthropic/claude-sonnet-4.6",
            messages=[{"role": "user", "content": "What is the weather in San Francisco?"}],
            response_model=Temperature,
            tools=[get_current_weather],
            max_recursion=10,
        )
        assert isinstance(response.content, Temperature)

    def test_prose_fallback_forced_extraction(self):
        """Model that answers in prose still yields structure via forced extraction."""
        response = structured_completion(
            model="openrouter/anthropic/claude-sonnet-4.6",
            messages=[{"role": "user", "content": "Just say: it is 18 degrees in Berlin."}],
            response_model=Temperature,
            max_recursion=5,
        )
        assert isinstance(response.content, Temperature)


@GEMINI
class TestGeminiStandardJsonPath:
    @pytest.mark.asyncio
    async def test_gemini_tools_plus_response_model(self):
        """Gemini tools + response_model now go through the standard JSON path
        (the old two-call hack is deleted)."""
        response = await astructured_completion(
            model="gemini/gemini-2.0-flash",
            messages=[{"role": "user", "content": "What is the weather in San Francisco?"}],
            response_model=Temperature,
            tools=[get_current_weather],
            max_recursion=10,
        )
        assert isinstance(response.content, Temperature)
