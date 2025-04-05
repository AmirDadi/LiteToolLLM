"""
LiteCallLLM - A lightweight wrapper for LLM function calling and structured output validation

This package extends litellm with structured output validation and transparent tool calling.
"""

__version__ = "0.1.0"

# Import core functionality from litetoolllm and re-export
from litetoolllm.core import structured_completion, astructured_completion, UnifiedResponse
from litetoolllm.tools import Tool
from litetoolllm.errors import StructuredValidationError
from litetoolllm.utils import convert_tools_to_api_format

# Make these accessible directly from litecallllm
__all__ = [
    'structured_completion', 
    'astructured_completion', 
    'UnifiedResponse', 
    'Tool', 
    'StructuredValidationError',
    'convert_tools_to_api_format'
] 