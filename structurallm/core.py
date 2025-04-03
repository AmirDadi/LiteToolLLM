# structurallm/core.py
from typing import Type, Any, List, Optional, Callable
from pydantic import BaseModel
from litellm import completion, acompletion
from .errors import ModelCapabilityError, StructuredValidationError, FunctionExecutionError, MaxRecursionError
import json
import litellm.utils

class UnifiedResponse(BaseModel):
    content: Optional[Any] = None
    messages: List[Any] = []

def structured_completion(*, model: str, messages: List[dict],
                          response_model: Optional[Type[BaseModel]] = None,
                          tools: Optional[List[Callable]] = None,
                          max_recursion: int = 3,
                          **kwargs) -> UnifiedResponse:
    validate_model_capabilities(model, response_model, tools)
    raw_response = completion(model=model, messages=messages, tools=convert_tools_to_api_format(tools), response_format=response_model, **kwargs)


    messages, raw_response = _handle_tool_call_loop(kwargs, max_recursion, messages, model, raw_response,
                                                    response_model, tools)

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


def validate_model_capabilities(model, response_model, tools):
    supported_params = {"json_mode": litellm.supports_response_schema(model=model),
                        "function_calling": litellm.supports_function_calling(model=model)}
    if response_model and not supported_params.get("json_mode", False):
        raise ModelCapabilityError(f"Model {model} lacks JSON support but response_model required")
    if tools and not supported_params.get("function_calling", False):
        raise ModelCapabilityError(f"Model {model} lacks tool calling but tools provided")


def _handle_tool_call_loop(kwargs, max_recursion, messages, model, raw_response, response_model,
                           tools):
    recursion_depth = 0
    while get_tool_calls(raw_response) is not None:
        recursion_depth += 1
        if recursion_depth and recursion_depth >= max_recursion:
            raise MaxRecursionError("Max recursion error in tool calling")
        new_messages = handle_tool_calls(raw_response=raw_response, tools=tools)
        messages = [*messages, *new_messages]
        raw_response = completion(model=model, messages=messages, tools=convert_tools_to_api_format(tools),
                                  response_format=response_model, **kwargs)
    if get_content_from_raw_response(raw_response) is not None:
        messages.append({
            "role": "assistant",
            "content": get_content_from_raw_response(raw_response)
        })
    return messages, raw_response


def get_content_from_raw_response(raw_response):
    return raw_response.get('choices', [{}])[0].get('message', {}).get('content', '{}')


def handle_tool_calls(raw_response, tools):
    tool_calls = get_tool_calls(raw_response)
    function_mapping = get_function_mapping(tools)
    new_messages = []
    if tool_calls:
        new_messages.append(raw_response.choices[0].message)
        for tool_call in tool_calls:
            try:
                print(f"\nExecuting tool call\n{tool_call}")
                function_name = tool_call.function.name
                function_to_call = function_mapping.get(function_name, None)
                if function_to_call is None:
                    raise ValueError(f"Function {function_name} name mismatch in tool calling")
                function_args = json.loads(tool_call.function.arguments)
                function_response = function_to_call(**function_args)
                new_messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps(function_response) if isinstance(function_response, dict) else function_response,
                    }
                )
            except Exception as e:
                raise FunctionExecutionError(tool_call, str(e)) from e
        return new_messages



def get_tool_calls(raw_response):
    return raw_response.get('choices', [{}])[0].get('message', {}).get('tool_calls', None)


def convert_tools_to_api_format(tools):
    dict_tools = None
    if tools:
        dict_tools = [{
            "type": "function",
            "function": litellm.utils.function_to_dict(tool)
        } for tool in tools]
    return dict_tools

def get_function_mapping(tools):
    mapping = {}
    for tool in tools:
        mapping[tool.__name__] = tool
    return mapping

async def astructured_completion(*, model: str, messages: List[dict],
                                 response_model: Optional[Type[BaseModel]] = None,
                                 **kwargs) -> UnifiedResponse:
    raw_response = await acompletion(model=model, messages=messages, **kwargs)
    return UnifiedResponse(raw_history=[raw_response])
