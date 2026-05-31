# litetoolllm/output_modes.py
"""Output-mode routing for structured completion.

This module holds the small, pure pieces that decide *how* a structured
response is produced for a given model and validate the result. Keeping them
here (and free of network/I/O) makes the routing rules easy to read and to test
without any API access.

Routing principle: native JSON mode wins whenever the model supports it;
tool-output (a synthetic ``final_result`` tool) only fills the gap for models
that can call tools but lack JSON-schema support.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Type

import litellm
from pydantic import BaseModel

from .errors import ModelCapabilityError

FINAL_RESULT_TOOL_NAME = "final_result"


@dataclass
class Capabilities:
    """What a model can do, as far as structured output is concerned."""
    function_calling: bool
    json_mode: bool


def resolve_capabilities(model: str, model_capabilities: Optional[dict] = None) -> Capabilities:
    """Resolve model capabilities, letting a caller override LiteLLM detection.

    ``model_capabilities`` may contain ``"function_calling"`` and/or
    ``"json_mode"`` booleans. Any key it provides wins; any key it omits falls
    back to LiteLLM's own detection. This is the single escape hatch for models
    whose capabilities LiteLLM reports incorrectly (common via OpenRouter).
    """
    overrides = model_capabilities or {}

    if "function_calling" in overrides:
        function_calling = overrides["function_calling"]
    else:
        function_calling = litellm.supports_function_calling(model=model)

    if "json_mode" in overrides:
        json_mode = overrides["json_mode"]
    else:
        json_mode = litellm.supports_response_schema(model=model)

    return Capabilities(function_calling=bool(function_calling), json_mode=bool(json_mode))


class OutputMode(Enum):
    """How the final response is produced.

    NONE        - no structured output requested (plain completion or tool loop)
    JSON        - native JSON / response_format mode
    TOOL_OUTPUT - structured output delivered via the ``final_result`` tool
    """
    NONE = "none"
    JSON = "json"
    TOOL_OUTPUT = "tool_output"


def select_output_mode(response_model: Optional[Type[BaseModel]],
                       tools: Optional[list],
                       caps: Capabilities) -> OutputMode:
    """Decide which output mode to use (pure; raises for unsupported models).

    See the module docstring for the routing principle. Evaluation order:
      1. no response_model and no tools          -> NONE
      2. tools present but no function calling   -> ModelCapabilityError
      3. no response_model (tools only)          -> NONE (plain tool loop)
      4. JSON supported                          -> JSON
      5. function calling supported              -> TOOL_OUTPUT
      6. otherwise                               -> ModelCapabilityError
    """
    if response_model is None and not tools:
        return OutputMode.NONE

    if tools and not caps.function_calling:
        raise ModelCapabilityError("Model lacks tool calling but tools provided")

    if response_model is None:
        # tools present and function calling is supported
        return OutputMode.NONE

    if caps.json_mode:
        return OutputMode.JSON

    if caps.function_calling:
        return OutputMode.TOOL_OUTPUT

    raise ModelCapabilityError(
        "Model lacks both JSON output and tool calling but response_model required"
    )


def build_final_result_tool(response_model: Type[BaseModel]) -> dict:
    """Build the synthetic ``final_result`` tool from a Pydantic model.

    The model's JSON schema becomes the tool's parameters; the model "calls"
    this tool to deliver its structured answer.
    """
    return {
        "type": "function",
        "function": {
            "name": FINAL_RESULT_TOOL_NAME,
            "description": (
                "Return the final structured answer. Call this exactly once, "
                "with all fields populated, when you have everything you need."
            ),
            "parameters": response_model.model_json_schema(),
        },
    }


def parse_final_result(arguments: dict, response_model: Type[BaseModel]) -> BaseModel:
    """Validate ``final_result`` call arguments into a model instance."""
    return response_model.model_validate(arguments)
