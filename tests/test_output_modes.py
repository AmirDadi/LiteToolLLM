# tests/test_output_modes.py
"""Deterministic (no-API) tests for the output-mode routing layer."""
import json

import pytest
import litellm

from litetoolllm.errors import ModelCapabilityError, StructuredValidationError
from litetoolllm.models import Temperature
from litetoolllm.output_modes import (
    Capabilities,
    OutputMode,
    FINAL_RESULT_TOOL_NAME,
    resolve_capabilities,
    select_output_mode,
    build_final_result_tool,
    parse_final_result,
)
from litetoolllm.utils import find_final_result_call


# --- resolve_capabilities: override precedence over LiteLLM detection ---------
class TestResolveCapabilities:
    def test_override_wins_over_detection(self, monkeypatch):
        monkeypatch.setattr(litellm, "supports_function_calling", lambda model: False)
        monkeypatch.setattr(litellm, "supports_response_schema", lambda model: False)

        caps = resolve_capabilities(
            "some-model", {"function_calling": True, "json_mode": True}
        )
        assert caps == Capabilities(function_calling=True, json_mode=True)

    def test_partial_override_falls_back_per_key(self, monkeypatch):
        monkeypatch.setattr(litellm, "supports_function_calling", lambda model: True)
        monkeypatch.setattr(litellm, "supports_response_schema", lambda model: False)

        # Only override json_mode; function_calling falls back to detection (True).
        caps = resolve_capabilities("some-model", {"json_mode": True})
        assert caps == Capabilities(function_calling=True, json_mode=True)

    def test_no_override_uses_detection(self, monkeypatch):
        monkeypatch.setattr(litellm, "supports_function_calling", lambda model: True)
        monkeypatch.setattr(litellm, "supports_response_schema", lambda model: False)

        caps = resolve_capabilities("some-model", None)
        assert caps == Capabilities(function_calling=True, json_mode=False)


# --- select_output_mode: every row of the routing table ----------------------
class TestSelectOutputMode:
    def test_no_model_no_tools_is_none(self):
        caps = Capabilities(function_calling=False, json_mode=False)
        assert select_output_mode(None, None, caps) is OutputMode.NONE

    def test_tools_without_function_calling_raises(self):
        caps = Capabilities(function_calling=False, json_mode=True)
        with pytest.raises(ModelCapabilityError):
            select_output_mode(Temperature, [lambda: None], caps)

    def test_tools_only_no_model_is_none(self):
        caps = Capabilities(function_calling=True, json_mode=False)
        assert select_output_mode(None, [lambda: None], caps) is OutputMode.NONE

    def test_json_wins_when_available(self):
        # JSON mode wins even when tools + function calling are present.
        caps = Capabilities(function_calling=True, json_mode=True)
        assert select_output_mode(Temperature, [lambda: None], caps) is OutputMode.JSON
        assert select_output_mode(Temperature, None, caps) is OutputMode.JSON

    def test_tool_output_fills_the_gap(self):
        # function calling but no JSON schema support -> tool output.
        caps = Capabilities(function_calling=True, json_mode=False)
        assert select_output_mode(Temperature, None, caps) is OutputMode.TOOL_OUTPUT
        assert select_output_mode(Temperature, [lambda: None], caps) is OutputMode.TOOL_OUTPUT

    def test_no_json_no_tools_raises(self):
        caps = Capabilities(function_calling=False, json_mode=False)
        with pytest.raises(ModelCapabilityError):
            select_output_mode(Temperature, None, caps)


# --- build_final_result_tool -------------------------------------------------
class TestBuildFinalResultTool:
    def test_schema_shape(self):
        tool = build_final_result_tool(Temperature)
        assert tool["type"] == "function"
        assert tool["function"]["name"] == FINAL_RESULT_TOOL_NAME
        assert "description" in tool["function"]
        params = tool["function"]["parameters"]
        # Mirrors the pydantic model's JSON schema.
        assert params == Temperature.model_json_schema()
        assert set(params["properties"]) == {"location", "temperature"}


# --- parse_final_result ------------------------------------------------------
class TestParseFinalResult:
    def test_happy_path(self):
        result = parse_final_result(
            {"location": "SF", "temperature": "68F"}, Temperature
        )
        assert isinstance(result, Temperature)
        assert result.location == "SF"

    def test_invalid_arguments_raise(self):
        with pytest.raises(Exception):
            parse_final_result({"location": "SF"}, Temperature)  # missing temperature


# --- find_final_result_call --------------------------------------------------
class _FakeFn:
    def __init__(self, name, arguments=""):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, name, arguments=""):
        self.function = _FakeFn(name, arguments)


class _FakeResponse:
    """Minimal stand-in shaped like a litellm response dict for get_tool_calls."""
    def __init__(self, tool_calls):
        self._d = {"choices": [{"message": {"tool_calls": tool_calls}}]}

    def get(self, key, default=None):
        return self._d.get(key, default)


class TestFindFinalResultCall:
    def test_finds_final_result_among_user_tools(self):
        calls = [
            _FakeToolCall("get_current_weather", "{}"),
            _FakeToolCall(FINAL_RESULT_TOOL_NAME, '{"location": "SF", "temperature": "68F"}'),
        ]
        found = find_final_result_call(_FakeResponse(calls))
        assert found is not None
        assert found.function.name == FINAL_RESULT_TOOL_NAME

    def test_returns_none_when_absent(self):
        calls = [_FakeToolCall("get_current_weather", "{}")]
        assert find_final_result_call(_FakeResponse(calls)) is None

    def test_returns_none_when_no_tool_calls(self):
        assert find_final_result_call(_FakeResponse(None)) is None
