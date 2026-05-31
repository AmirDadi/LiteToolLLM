# litetoolllm/core.py
import logging
from typing import Type, Any, List, Optional
from pydantic import BaseModel
from litellm import completion, acompletion
from .errors import StructuredValidationError
from .output_modes import (
    OutputMode,
    resolve_capabilities,
    select_output_mode,
    build_final_result_tool,
)
from .utils import (
    get_content_from_raw_response,
    get_usage_and_cost,
    _handle_tool_call_loop,
    _handle_tool_call_loop_async,
    run_tool_output_loop,
    run_tool_output_loop_async,
    convert_tools_to_api_format,
)

logger = logging.getLogger(__name__)

class UnifiedResponse(BaseModel):
    content: Optional[Any] = None
    messages: List[Any] = []
    # Observability fields populated from the final model response.
    # Default to None so existing callers are unaffected.
    usage: Optional[Any] = None
    cost: Optional[float] = None


def structured_completion(*, model: str, messages: List[dict],
                          response_model: Optional[Type[BaseModel]] = None,
                          tools: Optional[List] = None,
                          max_recursion: int = 3,
                          metadata=None,
                          model_capabilities: Optional[dict] = None,
                          **kwargs) -> UnifiedResponse:
    caps = resolve_capabilities(model, model_capabilities)
    mode = select_output_mode(response_model, tools, caps)

    if mode is OutputMode.TOOL_OUTPUT:
        return _run_tool_output_mode(
            model=model, messages=messages, response_model=response_model,
            tools=tools, max_recursion=max_recursion, metadata=metadata, **kwargs)

    return _run_json_mode(
        model=model, messages=messages, response_model=response_model,
        tools=tools, max_recursion=max_recursion, metadata=metadata, **kwargs)


def _run_json_mode(*, model, messages, response_model, tools, max_recursion,
                   metadata, **kwargs) -> UnifiedResponse:
    """Native JSON / response_format path (also covers the no-structure case)."""
    raw_response = completion(
        model=model,
        messages=messages,
        tools=convert_tools_to_api_format(tools),
        response_format=response_model,
        metadata=metadata,
        **kwargs
    )

    messages, raw_response = _handle_tool_call_loop(
        kwargs=kwargs,
        max_recursion=max_recursion,
        messages=messages,
        model=model,
        raw_response=raw_response,
        response_model=response_model,
        tools=tools,
        metadata=metadata,
    )

    response_content = get_content_from_raw_response(raw_response)
    try:
        if response_model:
            parsed = response_model.model_validate_json(response_content)
        else:
            parsed = response_content
    except Exception as e:
        logger.warning("Failed to validate response against %s: %s", response_model, e)
        raise StructuredValidationError("Failed to validate response", retry_context=raw_response) from e

    usage, cost = get_usage_and_cost(raw_response)
    return UnifiedResponse(content=parsed, messages=messages, usage=usage, cost=cost)


def _run_tool_output_mode(*, model, messages, response_model, tools, max_recursion,
                          metadata, **kwargs) -> UnifiedResponse:
    """Result-function path: the model delivers structure via ``final_result``."""
    raw_response = completion(
        model=model,
        messages=messages,
        tools=_tool_output_payload(tools, response_model),
        tool_choice="auto",
        metadata=metadata,
        **kwargs
    )

    messages, parsed, raw_response = run_tool_output_loop(
        kwargs=kwargs,
        max_recursion=max_recursion,
        messages=messages,
        model=model,
        raw_response=raw_response,
        response_model=response_model,
        tools=tools,
        metadata=metadata,
    )

    usage, cost = get_usage_and_cost(raw_response)
    return UnifiedResponse(content=parsed, messages=messages, usage=usage, cost=cost)


async def astructured_completion(*, model: str, messages: List[dict],
                                 response_model: Optional[Type[BaseModel]] = None,
                                 tools: Optional[List] = None,
                                 max_recursion: int = 3,
                                 metadata=None,
                                 model_capabilities: Optional[dict] = None,
                                 **kwargs) -> UnifiedResponse:
    caps = resolve_capabilities(model, model_capabilities)
    mode = select_output_mode(response_model, tools, caps)

    if mode is OutputMode.TOOL_OUTPUT:
        return await _run_tool_output_mode_async(
            model=model, messages=messages, response_model=response_model,
            tools=tools, max_recursion=max_recursion, metadata=metadata, **kwargs)

    return await _run_json_mode_async(
        model=model, messages=messages, response_model=response_model,
        tools=tools, max_recursion=max_recursion, metadata=metadata, **kwargs)


async def _run_json_mode_async(*, model, messages, response_model, tools, max_recursion,
                               metadata, **kwargs) -> UnifiedResponse:
    raw_response = await acompletion(
        model=model,
        messages=messages,
        tools=convert_tools_to_api_format(tools),
        response_format=response_model,
        metadata=metadata,
        **kwargs
    )
    messages, raw_response = await _handle_tool_call_loop_async(
        kwargs=kwargs,
        max_recursion=max_recursion,
        messages=messages,
        model=model,
        raw_response=raw_response,
        response_model=response_model,
        tools=tools,
        metadata=metadata,
    )

    response_content = get_content_from_raw_response(raw_response)
    try:
        if response_model:
            parsed = response_model.model_validate_json(response_content)
        else:
            parsed = response_content
    except Exception as e:
        logger.warning("Failed to validate response against %s: %s", response_model, e)
        raise StructuredValidationError("Failed to validate response", retry_context=raw_response) from e

    usage, cost = get_usage_and_cost(raw_response)
    return UnifiedResponse(content=parsed, messages=messages, usage=usage, cost=cost)


async def _run_tool_output_mode_async(*, model, messages, response_model, tools, max_recursion,
                                      metadata, **kwargs) -> UnifiedResponse:
    raw_response = await acompletion(
        model=model,
        messages=messages,
        tools=_tool_output_payload(tools, response_model),
        tool_choice="auto",
        metadata=metadata,
        **kwargs
    )

    messages, parsed, raw_response = await run_tool_output_loop_async(
        kwargs=kwargs,
        max_recursion=max_recursion,
        messages=messages,
        model=model,
        raw_response=raw_response,
        response_model=response_model,
        tools=tools,
        metadata=metadata,
    )

    usage, cost = get_usage_and_cost(raw_response)
    return UnifiedResponse(content=parsed, messages=messages, usage=usage, cost=cost)


def _tool_output_payload(tools, response_model):
    """User tools (if any) plus the synthetic ``final_result`` tool."""
    return (convert_tools_to_api_format(tools) or []) + [build_final_result_tool(response_model)]
