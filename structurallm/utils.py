import json
import litellm.utils
from litellm import acompletion, completion
from .errors import ModelCapabilityError, FunctionExecutionError, MaxRecursionError
import asyncio

def convert_tools_to_api_format(tools):
    dict_tools = None
    if tools:
        dict_tools = [{
            "type": "function",
            "function": litellm.utils.function_to_dict(tool)
        } for tool in tools]
    return dict_tools

def validate_model_capabilities(model, response_model, tools):
    supported_params = {"json_mode": litellm.supports_response_schema(model=model),
                        "function_calling": litellm.supports_function_calling(model=model)}
    if response_model and not supported_params.get("json_mode", False):
        raise ModelCapabilityError(f"Model {model} lacks JSON support but response_model required")
    if tools and not supported_params.get("function_calling", False):
        raise ModelCapabilityError(f"Model {model} lacks tool calling but tools provided")

def get_content_from_raw_response(raw_response):
    return raw_response.get('choices', [{}])[0].get('message', {}).get('content', '{}')

def get_tool_calls(raw_response):
    return raw_response.get('choices', [{}])[0].get('message', {}).get('tool_calls', None)

def get_function_mapping(tools):
    mapping = {}
    for tool in tools:
        mapping[tool.__name__] = tool
    return mapping

def _extract_function_details(tool_call, function_mapping):
    function_name = tool_call.function.name
    function_to_call = function_mapping.get(function_name, None)
    if function_to_call is None:
        raise ValueError(f"Function {function_name} name mismatch in tool calling")
    function_args = json.loads(tool_call.function.arguments)
    return function_name, function_to_call, function_args

def handle_tool_calls(raw_response, tools):
    tool_calls = get_tool_calls(raw_response)
    function_mapping = get_function_mapping(tools)
    new_messages = []
    if tool_calls:
        new_messages.append(raw_response.choices[0].message)
        for tool_call in tool_calls:
            try:
                print(f"\nExecuting tool call\n{tool_call}")
                function_name, function_to_call, function_args = _extract_function_details(tool_call, function_mapping)
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

async def handle_tool_calls_async(raw_response, tools):
    tool_calls = get_tool_calls(raw_response)
    if not tool_calls:
        return []



    async def execute_tool_call(tool_call, tools):
        function_mapping = get_function_mapping(tools)
        function_name, function_to_call, function_args = _extract_function_details(tool_call, function_mapping)
        result = await function_to_call(**function_args)
        
        return {
            "role": "tool",
            "tool_call_id": tool_call.get("id"),
            "content": json.dumps(result),
            "name": function_name
        }

    tasks = [execute_tool_call(tool_call, tools) for tool_call in tool_calls]
    responses = await asyncio.gather(*tasks)
    messages = [raw_response.choices[0].message, *responses]

    return messages

async def _handle_tool_call_loop_async(kwargs, max_recursion, messages, model, raw_response, response_model,
                           tools):
    recursion_depth = 0
    while get_tool_calls(raw_response) is not None:
        recursion_depth += 1
        if recursion_depth and recursion_depth >= max_recursion:
            raise MaxRecursionError("Max recursion error in tool calling")
        new_messages = await handle_tool_calls_async(raw_response=raw_response, tools=tools)
        messages = [*messages, *new_messages]
        raw_response = await acompletion(model=model, messages=messages, tools=convert_tools_to_api_format(tools),
                                  response_format=response_model, **kwargs)
    if get_content_from_raw_response(raw_response) is not None:
        messages.append({
            "role": "assistant",
            "content": get_content_from_raw_response(raw_response)
        })
    return messages, raw_response 