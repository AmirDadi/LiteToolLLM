# litetoolllm/__init__.py
from .core import structured_completion, astructured_completion, UnifiedResponse
from .tools import Tool
from .errors import StructuredValidationError
from .models import *
from .utils import convert_tools_to_api_format
from .conversation import saveConversation, trimHistory

__all__ = [
    'structured_completion',
    'astructured_completion',
    'UnifiedResponse',
    'Tool',
    'StructuredValidationError',
    'convert_tools_to_api_format',
    'saveConversation',
    'trimHistory'
]
