import json
import logging
import litellm.utils
from litellm import acompletion, completion
from .errors import (
    ModelCapabilityError,
    FunctionExecutionError,
    MaxRecursionError,
    StructuredValidationError,
)
from .output_modes import (
    FINAL_RESULT_TOOL_NAME,
    build_final_result_tool,
    parse_final_result,
)
import asyncio
import inspect

logger = logging.getLogger(__name__)

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

def get_usage_and_cost(raw_response):
    """Best-effort extraction of token usage and dollar cost from a model
    response. Returns ``(usage, cost)`` where either may be ``None`` if the
    information is unavailable (e.g. an unknown/custom model)."""
    usage = None
    cost = None
    try:
        usage = raw_response.get('usage')
    except Exception:
        logger.debug("Could not read usage from response", exc_info=True)
    try:
        cost = litellm.completion_cost(completion_response=raw_response)
    except Exception:
        logger.debug("Could not compute completion cost", exc_info=True)
    return usage, cost

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
                logger.debug("Executing tool call: %s", tool_call)
                function_name, function_to_call, function_args = _extract_function_details(tool_call, function_mapping)
                # Always remove metadata from LLM-provided args (the LLM may echo it back
                # because it's in the schema), then re-inject our own value if the function wants it.
                function_args.pop('metadata', None)
                sig = inspect.signature(function_to_call)
                if 'metadata' in sig.parameters:
                    function_response = function_to_call(**function_args, metadata=metadata)
                else:
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
                name = getattr(getattr(tool_call, "function", None), "name", "unknown")
                raise FunctionExecutionError(name, str(e), tool_call=tool_call) from e
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

    function_mapping = get_function_mapping(tools)

    async def execute_tool_call(tool_call):
        try:
            logger.debug("Executing tool call: %s", tool_call)
            function_name, function_to_call, function_args = _extract_function_details(tool_call, function_mapping)

            # Always remove metadata from LLM-provided args (the LLM may echo it back
            # because it's in the schema), then re-inject our own value if the function wants it.
            function_args.pop('metadata', None)
            sig = inspect.signature(function_to_call)
            accepts_metadata = 'metadata' in sig.parameters
            if inspect.iscoroutinefunction(function_to_call):
                result = await function_to_call(**function_args, metadata=metadata) if accepts_metadata else await function_to_call(**function_args)
            else:
                result = function_to_call(**function_args, metadata=metadata) if accepts_metadata else function_to_call(**function_args)

            return {
                "role": "tool",
                "tool_call_id": tool_call.get("id"),
                "content": json.dumps(result) if isinstance(result, dict) else result,
                "name": function_name
            }
        except Exception as e:
            name = getattr(getattr(tool_call, "function", None), "name", "unknown")
            raise FunctionExecutionError(name, str(e), tool_call=tool_call) from e

    tasks = [execute_tool_call(tool_call) for tool_call in tool_calls]
    responses = await asyncio.gather(*tasks)
    messages = [raw_response.choices[0].message.model_dump(), *responses]

    return messages

async def _handle_tool_call_loop_async(kwargs, max_recursion, messages, model, raw_response, response_model,
                           metadata, tools):
    recursion_depth = 0
    while get_tool_calls(raw_response) is not None:
        recursion_depth += 1
        if recursion_depth and recursion_depth >= max_recursion:
            raise MaxRecursionError("Max recursion error in tool calling")
        new_messages = await handle_tool_calls_async(raw_response=raw_response, tools=tools, metadata=metadata)
        messages = [*messages, *new_messages]
        raw_response = await acompletion(model=model, messages=messages, tools=convert_tools_to_api_format(tools),
                                  response_format=response_model, metadata=metadata, **kwargs)
    if get_content_from_raw_response(raw_response) is not None:
        messages.append({
            "role": "assistant",
            "content": get_content_from_raw_response(raw_response)
        })
    return messages, raw_response


# ---------------------------------------------------------------------------
# Tool-output (result-function) mode
# ---------------------------------------------------------------------------
# In this mode the response_model is registered as a synthetic ``final_result``
# tool. The model delivers its structured answer by "calling" that tool; we
# detect the call, validate its arguments, and stop. User tools (if any) run on
# the normal function-calling channel alongside it.


def find_final_result_call(raw_response):
    """Return the ``final_result`` tool call in a response, or ``None``.

    ``final_result`` is the synthetic tool that carries the structured answer,
    so it is handled specially rather than executed like a user function.
    """
    tool_calls = get_tool_calls(raw_response) or []
    for tool_call in tool_calls:
        if tool_call.function.name == FINAL_RESULT_TOOL_NAME:
            return tool_call
    return None


def _force_final_result_tool_choice():
    """tool_choice value that forces the model to call ``final_result``."""
    return {"type": "function", "function": {"name": FINAL_RESULT_TOOL_NAME}}


# A user-turn prompt for the prose-fallback case. It must end the conversation
# with a user message: some providers (e.g. Anthropic) reject a forced tool call
# when the last message is from the assistant ("assistant message prefill").
_FORCE_FINAL_RESULT_PROMPT = (
    f"Now return the structured answer by calling the {FINAL_RESULT_TOOL_NAME} "
    "tool with all fields populated."
)


def _append_forced_extraction_prompt(messages, raw_response):
    """Record the model's prose answer, then add the forcing user turn."""
    messages.append({"role": "assistant", "content": get_content_from_raw_response(raw_response)})
    messages.append({"role": "user", "content": _FORCE_FINAL_RESULT_PROMPT})


def _parse_final_result_call(final_call, response_model, raw_response):
    """Validate a ``final_result`` tool call's arguments into a model instance.

    Validation failures surface as ``StructuredValidationError`` (matching JSON
    mode), carrying the raw response as ``retry_context``.
    """
    try:
        arguments = json.loads(final_call.function.arguments)
        return parse_final_result(arguments, response_model)
    except StructuredValidationError:
        raise
    except Exception as e:
        logger.warning("final_result arguments failed validation: %s", e)
        raise StructuredValidationError(
            "Failed to validate final_result arguments", retry_context=raw_response
        ) from e


def run_tool_output_loop(kwargs, max_recursion, messages, model, raw_response,
                         response_model, tools, metadata):
    """Drive the synchronous ``final_result`` tool-output path.

    Loops on user tool calls until the model calls ``final_result``. If the
    model instead answers in prose, a single forced-extraction call asks it to
    populate ``final_result``. Returns ``(messages, parsed, raw_response)``.
    """
    tool_payload = (convert_tools_to_api_format(tools) or []) + [
        build_final_result_tool(response_model)
    ]
    recursion_depth = 0
    while True:
        final_call = find_final_result_call(raw_response)
        if final_call is not None:
            messages.append(raw_response.choices[0].message)
            parsed = _parse_final_result_call(final_call, response_model, raw_response)
            return messages, parsed, raw_response

        if get_tool_calls(raw_response) is None:
            break  # model answered without tools; force extraction below

        recursion_depth += 1
        if recursion_depth >= max_recursion:
            raise MaxRecursionError("Max recursion error in tool calling")
        new_messages = handle_tool_calls(raw_response=raw_response, tools=tools, metadata=metadata)
        messages = [*messages, *new_messages]
        raw_response = completion(model=model, messages=messages, tools=tool_payload,
                                  tool_choice="auto", metadata=metadata, **kwargs)

    # Prose fallback: one forced extraction call.
    _append_forced_extraction_prompt(messages, raw_response)
    raw_response = completion(model=model, messages=messages,
                              tools=[build_final_result_tool(response_model)],
                              tool_choice=_force_final_result_tool_choice(),
                              metadata=metadata, **kwargs)
    final_call = find_final_result_call(raw_response)
    if final_call is None:
        raise StructuredValidationError(
            "Model did not return final_result", retry_context=raw_response
        )
    messages.append(raw_response.choices[0].message)
    parsed = _parse_final_result_call(final_call, response_model, raw_response)
    return messages, parsed, raw_response


async def run_tool_output_loop_async(kwargs, max_recursion, messages, model, raw_response,
                                     response_model, tools, metadata):
    """Async mirror of :func:`run_tool_output_loop`."""
    tool_payload = (convert_tools_to_api_format(tools) or []) + [
        build_final_result_tool(response_model)
    ]
    recursion_depth = 0
    while True:
        final_call = find_final_result_call(raw_response)
        if final_call is not None:
            messages.append(raw_response.choices[0].message.model_dump())
            parsed = _parse_final_result_call(final_call, response_model, raw_response)
            return messages, parsed, raw_response

        if get_tool_calls(raw_response) is None:
            break

        recursion_depth += 1
        if recursion_depth >= max_recursion:
            raise MaxRecursionError("Max recursion error in tool calling")
        new_messages = await handle_tool_calls_async(raw_response=raw_response, tools=tools, metadata=metadata)
        messages = [*messages, *new_messages]
        raw_response = await acompletion(model=model, messages=messages, tools=tool_payload,
                                         tool_choice="auto", metadata=metadata, **kwargs)

    _append_forced_extraction_prompt(messages, raw_response)
    raw_response = await acompletion(model=model, messages=messages,
                                     tools=[build_final_result_tool(response_model)],
                                     tool_choice=_force_final_result_tool_choice(),
                                     metadata=metadata, **kwargs)
    final_call = find_final_result_call(raw_response)
    if final_call is None:
        raise StructuredValidationError(
            "Model did not return final_result", retry_context=raw_response
        )
    messages.append(raw_response.choices[0].message.model_dump())
    parsed = _parse_final_result_call(final_call, response_model, raw_response)
    return messages, parsed, raw_response