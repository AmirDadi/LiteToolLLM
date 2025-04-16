# litetoolllm/core.py
from typing import Type, Any, List, Optional, Callable
from pydantic import BaseModel
from litellm import completion, acompletion
from .errors import StructuredValidationError
from .utils import (
    validate_model_capabilities,
    get_content_from_raw_response,
    _handle_tool_call_loop,
    _handle_tool_call_loop_async,
    convert_tools_to_api_format,
)

class UnifiedResponse(BaseModel):
    content: Optional[Any] = None
    messages: List[Any] = []

def structured_completion(*, model: str, messages: List[dict],
                          response_model: Optional[Type[BaseModel]] = None,
                          tools: Optional[List] = None,
                          max_recursion: int = 3,
                          metadata=None,
                          **kwargs) -> UnifiedResponse:
    validate_model_capabilities(model, response_model, tools)
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
    parsed = None
    try:
        if response_model:
            parsed = response_model.parse_raw(response_content)
        else:
            parsed = response_content
    except Exception as e:
        raise StructuredValidationError("Failed to validate response", retry_context=raw_response) from e

    return UnifiedResponse(
        content=parsed,
        messages=messages
    )

async def astructured_completion(*, model: str, messages: List[dict],
                                 response_model: Optional[Type[BaseModel]] = None,
                                 tools: Optional[List] = None,
                                 max_recursion: int = 3,
                                 metadata = None,
                                 **kwargs) -> UnifiedResponse:
    # validate_model_capabilities(model, response_model, tools)
    raw_response = await acompletion(
        model=model,
        messages=messages,
        tools=convert_tools_to_api_format(tools),
        response_format=response_model,
        metadata=metadata,
        **kwargs
    )
    # metadata.pop('trace_id', None)
    messages, raw_response = await _handle_tool_call_loop_async(
        kwargs=kwargs,
        max_recursion=max_recursion,
        messages=messages,
        model=model,
        raw_response=raw_response,
        response_model=response_model,
        tools=tools,
        metadata=metadata
    )

    response_content = get_content_from_raw_response(raw_response)
    parsed = None
    try:
        if response_model:
            parsed = response_model.parse_raw(response_content)
        else:
            parsed = response_content
    except Exception as e:
        raise StructuredValidationError("Failed to validate response", retry_context=raw_response) from e

    return UnifiedResponse(
        content=parsed,
        messages=messages
    )
