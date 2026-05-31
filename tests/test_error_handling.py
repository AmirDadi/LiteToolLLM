# tests/test_error_handling.py
"""Deterministic (no-API) tests for the error hierarchy and observability fields.

These lock in:
- every library error inheriting from `LiteToolLLMError`
- `FunctionExecutionError` carrying the tool name, details and raw tool_call
- tool failures (sync and async) being surfaced as `FunctionExecutionError`
- `UnifiedResponse` exposing optional `usage`/`cost` fields
- `get_usage_and_cost` degrading gracefully

They run without any API key.
"""
import pytest

from litetoolllm.core import UnifiedResponse
from litetoolllm.errors import (
    LiteToolLLMError,
    ModelCapabilityError,
    MaxRecursionError,
    StructuredValidationError,
    FunctionExecutionError,
    RecursionDepthExceedError,
)
from litetoolllm.utils import handle_tool_calls, handle_tool_calls_async, get_usage_and_cost

from test_metadata_handling import _ToolCall, _ToolCallAsync, _sync_raw, _async_raw


# --- Exception hierarchy ----------------------------------------------------

class TestExceptionHierarchy:
    @pytest.mark.parametrize("err_cls", [
        ModelCapabilityError,
        MaxRecursionError,
        StructuredValidationError,
        FunctionExecutionError,
        RecursionDepthExceedError,
    ])
    def test_all_errors_inherit_base(self, err_cls):
        assert issubclass(err_cls, LiteToolLLMError)

    def test_function_execution_error_attributes(self):
        err = FunctionExecutionError("weather", "boom", tool_call={"id": "call_1"})
        assert err.function_name == "weather"
        assert err.details == "boom"
        assert err.tool_call == {"id": "call_1"}
        assert str(err) == "Error in weather: boom"

    def test_function_execution_error_tool_call_defaults_none(self):
        err = FunctionExecutionError("weather", "boom")
        assert err.tool_call is None

    def test_structured_validation_error_keeps_retry_context(self):
        err = StructuredValidationError("bad", retry_context={"raw": 1})
        assert err.retry_context == {"raw": 1}


# --- Tool failures surface as FunctionExecutionError ------------------------

class TestSyncToolFailure:
    def test_failing_tool_raises_function_execution_error(self):
        def weather(location: str) -> dict:
            raise RuntimeError("upstream down")

        tc = _ToolCall("call_1", "weather", '{"location": "SF"}')
        with pytest.raises(FunctionExecutionError) as exc:
            handle_tool_calls(raw_response=_sync_raw([tc]), tools=[weather], metadata=None)

        assert exc.value.function_name == "weather"
        assert exc.value.tool_call is tc
        assert isinstance(exc.value.__cause__, RuntimeError)


class TestAsyncToolFailure:
    async def test_failing_async_tool_raises_function_execution_error(self):
        async def weather(location: str) -> dict:
            raise RuntimeError("upstream down")

        tc = _ToolCallAsync("call_1", "weather", '{"location": "SF"}')
        with pytest.raises(FunctionExecutionError) as exc:
            await handle_tool_calls_async(raw_response=_async_raw([tc]), tools=[weather], metadata=None)

        assert exc.value.function_name == "weather"
        assert exc.value.tool_call is tc
        assert isinstance(exc.value.__cause__, RuntimeError)


# --- Observability ----------------------------------------------------------

class TestUnifiedResponseObservability:
    def test_usage_and_cost_default_to_none(self):
        resp = UnifiedResponse(content={"ok": True}, messages=[])
        assert resp.usage is None
        assert resp.cost is None

    def test_usage_and_cost_can_be_set(self):
        resp = UnifiedResponse(content=None, messages=[], usage={"total_tokens": 42}, cost=0.001)
        assert resp.usage == {"total_tokens": 42}
        assert resp.cost == 0.001


class TestGetUsageAndCost:
    def test_reads_usage_and_never_raises(self):
        usage, cost = get_usage_and_cost({"usage": {"total_tokens": 10}})
        assert usage == {"total_tokens": 10}
        # cost may be None (unknown/fake model) but the call must not raise.

    def test_missing_usage_returns_none(self):
        usage, cost = get_usage_and_cost({})
        assert usage is None
