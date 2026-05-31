# litetoolllm/conversation.py
"""Helpers for persisting and trimming conversation history."""
import json
import logging
import os
from typing import List

logger = logging.getLogger(__name__)


def save_conversation(messages: List[dict], directory: str, name: str) -> str:
    """Persist a conversation's messages to ``<directory>/<name>.json``.

    ``name`` is a caller-supplied conversation identifier (for example a
    session id taken from an incoming request). Returns the written path.
    """
    os.makedirs(directory, exist_ok=True)
    safe_name = os.path.normpath(name)
    path = os.path.join(directory, f"{safe_name}.json")
    with open(path, "w") as f:
        json.dump(messages, f)
    logger.debug("Saved conversation to %s", path)
    return path


def trim_history(messages: List[dict], max_messages: int) -> List[dict]:
    """Trim history to the most recent ``max_messages`` messages, always
    keeping a leading system message if present."""
    if len(messages) <= max_messages:
        return messages
    if messages and messages[0].get("role") == "system":
        return [messages[0], *messages[-(max_messages - 1):]]
    return messages[-max_messages:]
