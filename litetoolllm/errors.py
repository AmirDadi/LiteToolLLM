# litetoolllm/errors.py


class LiteToolLLMError(Exception):
    """Base class for every error raised by litetoolllm.

    Catch this to handle any library error in one place:

        try:
            structured_completion(...)
        except LiteToolLLMError:
            ...
    """
    pass


class ModelCapabilityError(LiteToolLLMError):
    """The requested model does not support a required capability
    (JSON/structured output or tool calling)."""
    pass


class MaxRecursionError(LiteToolLLMError):
    """The tool-calling loop hit ``max_recursion`` without producing a
    final answer."""
    pass


class StructuredValidationError(LiteToolLLMError):
    """The model response could not be validated against ``response_model``.

    ``retry_context`` holds the raw model response that failed validation.
    """
    def __init__(self, message, retry_context=None):
        super().__init__(message)
        self.retry_context = retry_context


class FunctionExecutionError(LiteToolLLMError):
    """A tool raised while being executed during a tool call.

    Attributes:
        function_name: Name of the tool that failed.
        details: String form of the underlying error.
        tool_call: The raw tool-call object from the model (if available).

    The original exception is preserved as ``__cause__`` (``raise ... from e``).
    """
    def __init__(self, function_name, details, tool_call=None):
        super().__init__(f"Error in {function_name}: {details}")
        self.function_name = function_name
        self.details = details
        self.tool_call = tool_call


class RecursionDepthExceedError(LiteToolLLMError):
    """Deprecated: kept for backward compatibility; not raised internally.
    Use :class:`MaxRecursionError` instead."""
    pass
