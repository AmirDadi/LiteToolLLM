# tests/test_metadata_handling.py
"""Deterministic (no-API) tests for tool metadata handling.

These lock in the P0 fix that lets tool functions which do NOT declare a
`metadata` parameter be called correctly, while functions that DO declare it
receive the caller-injected value (never the value the LLM may echo back from
the schema). They run without any API key.
"""
import json
import pytest

from litetoolllm.utils import handle_tool_calls, handle_tool_calls_async
from litetoolllm.tools import Tool


# --- Fakes that mimic the litellm response/tool_call shape ------------------
# The library accesses the raw response both as a dict (get_tool_calls) and via
# attributes (raw_response.choices[0].message), so the fakes support both.

class _Function:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    """Sync tool_call: attribute access for .function / .id."""
    def __init__(self, id, name, arguments):
        self.id = id
        self.function = _Function(name, arguments)


class _ToolCallAsync(dict):
    """Async tool_call: attribute access for .function plus dict .get('id')."""
    def __init__(self, id, name, arguments):
        super().__init__(id=id)
        self.function = _Function(name, arguments)


class _Message:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls
        self.role = "assistant"

    def model_dump(self):
        return {"role": "assistant", "content": None}


class _Choice:
    def __init__(self, message):
        self.message = message


class _RawResponse(dict):
    def __init__(self, tool_calls):
        super().__init__(choices=[{"message": {"tool_calls": tool_calls, "content": None}}])
        self.choices = [_Choice(_Message(tool_calls))]


def _sync_raw(tool_calls):
    return _RawResponse(tool_calls)


def _async_raw(tool_calls):
    return _RawResponse(tool_calls)


# --- Sync -------------------------------------------------------------------

class TestSyncMetadataHandling:
    def test_tool_without_metadata_param_is_called(self):
        """A plain tool with no `metadata` param runs without error."""
        def weather(location: str) -> dict:
            return {"location": location, "temperature": "68F"}

        tc = _ToolCall("call_1", "weather", '{"location": "SF"}')
        msgs = handle_tool_calls(raw_response=_sync_raw([tc]), tools=[weather], metadata={"k": "v"})

        assert msgs[1]["role"] == "tool"
        assert json.loads(msgs[1]["content"]) == {"location": "SF", "temperature": "68F"}

    def test_tool_with_metadata_param_receives_injected_value(self):
        """Caller-injected metadata wins; LLM-echoed metadata in args is dropped."""
        captured = {}

        def weather(location: str, metadata: dict) -> dict:
            captured["metadata"] = metadata
            return {"location": location}

        # The LLM echoes metadata back in the arguments; it must be ignored.
        tc = _ToolCall("call_1", "weather", '{"location": "SF", "metadata": {"from_llm": true}}')
        handle_tool_calls(raw_response=_sync_raw([tc]), tools=[weather], metadata={"from_caller": 1})

        assert captured["metadata"] == {"from_caller": 1}

    def test_tool_class_without_metadata_param(self):
        """Tool wrapper unwraps to .func and runs a metadata-free function."""
        def weather(location: str) -> dict:
            return {"location": location}

        tool = Tool(func=weather, name="get_weather", description="desc")
        tc = _ToolCall("call_1", "get_weather", '{"location": "SF"}')
        msgs = handle_tool_calls(raw_response=_sync_raw([tc]), tools=[tool], metadata=None)

        assert json.loads(msgs[1]["content"]) == {"location": "SF"}


# --- Async ------------------------------------------------------------------

class TestAsyncMetadataHandling:
    async def test_async_tool_without_metadata_param_is_called(self):
        async def weather(location: str) -> dict:
            return {"location": location, "temperature": "68F"}

        tc = _ToolCallAsync("call_1", "weather", '{"location": "SF"}')
        msgs = await handle_tool_calls_async(raw_response=_async_raw([tc]), tools=[weather], metadata={"k": "v"})

        assert msgs[1]["role"] == "tool"
        assert json.loads(msgs[1]["content"]) == {"location": "SF", "temperature": "68F"}

    async def test_async_tool_with_metadata_param_receives_injected_value(self):
        captured = {}

        async def weather(location: str, metadata: dict) -> dict:
            captured["metadata"] = metadata
            return {"location": location}

        tc = _ToolCallAsync("call_1", "weather", '{"location": "SF", "metadata": {"from_llm": true}}')
        await handle_tool_calls_async(raw_response=_async_raw([tc]), tools=[weather], metadata={"from_caller": 1})

        assert captured["metadata"] == {"from_caller": 1}

    async def test_async_loop_runs_plain_sync_tool_without_metadata(self):
        """The async loop must also handle a plain (sync) metadata-free tool."""
        def weather(location: str) -> dict:
            return {"location": location}

        tc = _ToolCallAsync("call_1", "weather", '{"location": "SF"}')
        msgs = await handle_tool_calls_async(raw_response=_async_raw([tc]), tools=[weather], metadata=None)

        assert json.loads(msgs[1]["content"]) == {"location": "SF"}
