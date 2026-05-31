# litetoolllm/core.py
import logging
from typing import Type, Any, List, Optional, Callable
from pydantic import BaseModel
from litellm import completion, acompletion
from .errors import StructuredValidationError
from .utils import (
    validate_model_capabilities,
    get_content_from_raw_response,
    get_usage_and_cost,
    _handle_tool_call_loop,
    _handle_tool_call_loop_async,
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
            parsed = response_model.model_validate_json(response_content)
        else:
            parsed = response_content
    except Exception as e:
        logger.warning("Failed to validate response against %s: %s", response_model, e)
        raise StructuredValidationError("Failed to validate response", retry_context=raw_response) from e

    usage, cost = get_usage_and_cost(raw_response)
    return UnifiedResponse(
        content=parsed,
        messages=messages,
        usage=usage,
        cost=cost,
    )

async def astructured_completion(*, model: str, messages: List[dict],
                                 response_model: Optional[Type[BaseModel]] = None,
                                 tools: Optional[List] = None,
                                 max_recursion: int = 3,
                                 metadata = None,
                                 **kwargs) -> UnifiedResponse:
    post_format_response_model = None
    if 'gemini' in model and tools and len(tools) > 0 and response_model is not None and tools[0].get("googleSearch") is None:
        post_format_response_model = response_model
        response_model = None
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
        post_format_response_model=post_format_response_model
    )

    response_content = get_content_from_raw_response(raw_response)
    parsed = None
    try:
        if response_model:
            parsed = response_model.model_validate_json(response_content)
        elif post_format_response_model:
            parsed = post_format_response_model.model_validate_json(response_content)
        else:
            parsed = response_content
    except Exception as e:
        logger.warning("Failed to validate response against %s: %s",
                       response_model or post_format_response_model, e)
        raise StructuredValidationError("Failed to validate response", retry_context=raw_response) from e

    usage, cost = get_usage_and_cost(raw_response)
    return UnifiedResponse(
        content=parsed,
        messages=messages,
        usage=usage,
        cost=cost,
    )
