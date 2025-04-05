# litetoolllm/errors.py
class ModelCapabilityError(Exception):
    pass


class MaxRecursionError(Exception):
    pass
class StructuredValidationError(Exception):
    def __init__(self, message, retry_context=None):
        super().__init__(message)
        self.retry_context = retry_context

class FunctionExecutionError(Exception):
    def __init__(self, function_name, details):
        super().__init__(f"Error in {function_name}: {details}")
        self.function_name = function_name
        self.details = details

class RecursionDepthExceedError(Exception):
    pass
