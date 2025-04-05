"""
LiteCallLLM - A lightweight wrapper for LLM function calling and structured output validation

This package extends litellm with structured output validation and transparent tool calling.
"""

__version__ = "0.1.0"

# Import key functionality to make it accessible at the top level
from structurallm.core import structured_completion, astructured_completion, UnifiedResponse
from structurallm.tools import Tool
from structurallm.errors import StructuredValidationError 