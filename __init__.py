"""
LiteCallLLM - A lightweight wrapper for LLM function calling and structured output validation

This package extends litellm with structured output validation and transparent tool calling.
"""

__version__ = "0.1.0"

# Import core functionality from structurallm and re-export
from litetoollm.core import structured_completion, astructured_completion, UnifiedResponse
from litetoollm.tools import Tool
from litetoollm.errors import StructuredValidationError
from litetoollm.utils import convert_tools_to_api_format

# Make these accessible directly from litecallllm
__all__ = [
    'structured_completion', 
    'astructured_completion', 
    'UnifiedResponse', 
    'Tool', 
    'StructuredValidationError',
    'convert_tools_to_api_format'
] 