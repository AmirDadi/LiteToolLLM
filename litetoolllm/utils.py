import json
import litellm.utils
from litellm import acompletion, completion
from .errors import ModelCapabilityError, FunctionExecutionError, MaxRecursionError
import asyncio
import inspect

structured_output_prompt = """
Make the output of last response structured. 
"""
def convert_tools_to_api_format(tools):
    if not tools:
        return None
    
    dict_tools = []
    for tool in tools:
        # Check if tool is already a callable function
        if callable(tool) and not hasattr(tool, 'func'):
            dict_tools.append({
                "type": "function",
                "function": litellm.utils.function_to_dict(tool)
            })
        # Check if tool is a Tool instance
        elif hasattr(tool, 'func') and callable(tool.func):
            # If it's a Tool instance, use the provided parameters or convert from the function
            if tool.parameters:
                function_dict = {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            else:
                function_dict = litellm.utils.function_to_dict(tool.func)
                function_dict["name"] = tool.name
                function_dict["description"] = tool.description
            
            dict_tools.append({
                "type": "function",
                "function": function_dict
            })
        else:
            dict_tools.append(tool)
    
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
        if callable(tool) and not hasattr(tool, 'func'):
            # Regular function
            mapping[tool.__name__] = tool
        elif hasattr(tool, 'func') and callable(tool.func):
            # Tool instance
            mapping[tool.name] = tool
    return mapping

def _extract_function_details(tool_call, function_mapping):
    function_name = tool_call.function.name
    function_to_call = function_mapping.get(function_name, None)
    if function_to_call is None:
        raise ValueError(f"Function {function_name} name mismatch in tool calling")
    
    function_args = json.loads(tool_call.function.arguments)
    
    # If it's a Tool instance, get the actual function
    if hasattr(function_to_call, 'func'):
        function_to_call = function_to_call.func
        
    return function_name, function_to_call, function_args

def handle_tool_calls(raw_response, tools, metadata):
    tool_calls = get_tool_calls(raw_response)
    function_mapping = get_function_mapping(tools)
    new_messages = []
    if tool_calls:
        new_messages.append(raw_response.choices[0].message)
        for tool_call in tool_calls:
            try:
                print(f"\nExecuting tool call\n{tool_call}")
                function_name, function_to_call, function_args = _extract_function_details(tool_call, function_mapping)
                function_response = function_to_call(**function_args, metadata=metadata)
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
                           tools, metadata):
    recursion_depth = 0
    while get_tool_calls(raw_response) is not None:
        recursion_depth += 1
        if recursion_depth and recursion_depth >= max_recursion:
            raise MaxRecursionError("Max recursion error in tool calling")
        new_messages = handle_tool_calls(raw_response=raw_response, tools=tools, metadata=metadata)
        messages = [*messages, *new_messages]
        raw_response = completion(model=model, messages=messages, tools=convert_tools_to_api_format(tools),
                                  response_format=response_model, metadata=metadata, **kwargs)
    if get_content_from_raw_response(raw_response) is not None:
        messages.append({
            "role": "assistant",
            "content": get_content_from_raw_response(raw_response)
        })
    return messages, raw_response

async def handle_tool_calls_async(raw_response, tools, metadata):
    tool_calls = get_tool_calls(raw_response)
    if not tool_calls:
        return []

    async def execute_tool_call(tool_call, tools):
        function_mapping = get_function_mapping(tools)
        function_name, function_to_call, function_args = _extract_function_details(tool_call, function_mapping)
        
        # Check if function is async
        function_args.pop('metadata', None)
        if inspect.iscoroutinefunction(function_to_call):
            result = await function_to_call(**function_args, metadata=metadata)
        else:
            result = function_to_call(**function_args, metadata=metadata)
        
        return {
            "role": "tool",
            "tool_call_id": tool_call.get("id"),
            "content": json.dumps(result) if isinstance(result, dict) else result,
            "name": function_name
        }

    tasks = [execute_tool_call(tool_call, tools) for tool_call in tool_calls]
    responses = await asyncio.gather(*tasks)
    messages = [raw_response.choices[0].message.model_dump(), *responses]

    return messages

async def _handle_tool_call_loop_async(kwargs, max_recursion, messages, model, raw_response, response_model,
                           metadata, tools, post_format_response_model=None):
    recursion_depth = 0
    while get_tool_calls(raw_response) is not None:
        recursion_depth += 1
        if recursion_depth and recursion_depth >= max_recursion:
            raise MaxRecursionError("Max recursion error in tool calling")
        new_messages = await handle_tool_calls_async(raw_response=raw_response, tools=tools, metadata=metadata)
        messages = [*messages, *new_messages]
        raw_response = await acompletion(model=model, messages=messages, tools=convert_tools_to_api_format(tools),
                                  response_format=response_model, metadata=metadata, **kwargs)
    if post_format_response_model:
        structured_output_messages = [
            *messages,
            {"role": "assistant", "content": get_content_from_raw_response(raw_response)},
            {"role": "system", "content": structured_output_prompt}
        ]
        raw_response = await acompletion(model="gemini/gemini-2.0-flash",
                                         messages=structured_output_messages, metadata=metadata, response_format=post_format_response_model)
    if get_content_from_raw_response(raw_response) is not None:
        messages.append({
            "role": "assistant",
            "content": get_content_from_raw_response(raw_response)
        })
    return messages, raw_response 