# LiteToolLLM

LiteToolLLM is a lightweight wrapper built on top of [litellm](https://github.com/BerriAI/litellm) that seamlessly integrates structured output validation with transparent tool calling for large language models. While litellm already provides powerful features such as synchronous (`completion`) and asynchronous (`acompletion`) completions, JSON-based response schemas, and built-in function calling capabilities, litetoolllm extends these functionalities with a clear, debuggable mechanism for directly invoking tools based on LLM outputs—without the complexity of traditional agent frameworks.

**Key Features:**

- **Transparent Tool Calling:**  
  Easily trace and debug function invocations directly from your LLM responses, giving you full visibility into which tools are executed and when.

- **Structured Output Validation:**  
  Use Pydantic models to ensure that responses conform to your expected schema, enhancing reliability and simplifying downstream processing.

- **Automated Tool Call Handling:**  
  No need to manually handle tool call responses or convert outputs. litetoolllm automatically processes tool responses, streamlining your workflow.

- **Full litellm Integration:**  
  Leverage all the features of litellm—including synchronous and asynchronous completions, JSON mode support, and function calling—while gaining additional control and transparency.

- **Minimalistic and Intuitive:**  
  Designed to avoid the overhead and complexity of frameworks like LangChain or LangGraph, litetoolllm offers a straightforward, powerful solution for advanced LLM interactions.
- **Recursion Handling**: 
  Configurable depth for complex tool chains
- **Sync/Async Support**:
  Full support for both synchronous and asynchronous operations with parallel tool execution capabilities.

## When to Use litetoolllm

litetoolllm is designed for a specific gap: **a single LLM call that may need tools, returning structured output** — without spinning up a full agent.

| | litetoolllm | pydantic-ai | LangChain |
|---|---|---|---|
| Setup | 3 lines | ~15 lines | ~20 lines |
| Use case | Single structured call with tools | Full agent loop | Complex pipelines |
| Provider coverage | All litellm providers (~100+) | ~10 providers | Many (varies) |
| Streaming | No | Yes | Yes |
| Multi-turn agent | Manual (pass `messages` back) | Built-in | Built-in |
| Overhead | Minimal | Low | High |

**Choose litetoolllm when** you want one clean function call that handles tool resolution and returns a typed result.  
**Choose pydantic-ai** when you need agent memory, retries, streaming, or dependency injection.

## Installation

You can install litetoolllm directly from the GitHub repository:

```bash
pip install git+https://github.com/AmirDadi/liteToolLlm.git
```

For development installation with additional testing dependencies:

```bash
pip install git+https://github.com/AmirDadi/liteToolLlm.git#egg=litetoolllm[dev]
```

## Model Compatibility

| Provider | Structured Output (`response_model`) | Tool Calling (`tools`) | Notes |
|----------|--------------------------------------|------------------------|-------|
| OpenAI (gpt-4o, gpt-4o-mini, etc.) | Yes | Yes | Full support |
| Google Gemini | Yes | Yes | Async recommended |
| Anthropic Claude | No* | Yes | Use tools-only |
| Mistral | Yes | Yes | |
| Groq | No | Yes | |

*Anthropic structured output via tool-calling workaround is not yet supported in this library.

litetoolllm delegates model capability checks to litellm — if `litellm.supports_response_schema(model)` returns true, structured output will be attempted.

## Usage Examples

### 1. Single Tool Execution

Invoke a tool (e.g., a weather lookup) and have the response validated against a structured schema.

```python
from litetoolllm import structured_completion
from pydantic import BaseModel

class Temperature(BaseModel):
    location: str
    temperature: str

def get_current_weather(location: str) -> dict:
    """Get the current weather in a given location"""
    if "San Francisco" in location:
        return {"location": "San Francisco", "temperature": "68°F"}
    return {"location": location, "temperature": "unknown"}

response = structured_completion(
    model="gemini/gemini-2.0-flash",
    messages=[{"role": "user", "content": "What is the weather in San Francisco?"}],
    response_model=Temperature,
    tools=[get_current_weather],
    max_recursion=10
)
```
### 2. Single Tool Execution without Schema
Use litetoolllm without providing a response model. The output will be returned as a plain string.

```python
response = structured_completion(
  model="gpt-4o-mini",
  messages=[{"role": "user", "content": "What is the weather in San Francisco?"}],
  tools=[get_current_weather],
  max_recursion=10
)
```

### 3. Multiple Tool Parallel Execution
Execute multiple tool calls in parallel and combine their outputs. This example shows how to handle responses from multiple locations.
```python
response = structured_completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is the weather in San Francisco and New York?"}],
    response_model=Temperatures,
    tools=[get_current_weather],
    max_recursion=10,
    parallel_tool_calls=True
)
```

### 4. Async Tool Execution
Execute tools asynchronously for better performance:

```python
response = await astructured_completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is the weather in San Francisco?"}],
    response_model=Temperature,
    tools=[get_current_weather],
    max_recursion=10
)
```

### 5. Async Parallel Tool Execution
Execute multiple tools in parallel asynchronously:

```python
response = await astructured_completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is the weather in San Francisco and New York?"}],
    response_model=Temperatures,
    tools=[get_current_weather],
    max_recursion=10,
    parallel_tool_calls=True
)
```

### 6. Custom Tool Name and Description

Use the `Tool` class to override a function's name, description, or parameter schema as seen by the LLM:

```python
from litetoolllm import structured_completion, Tool

def _internal_weather_lookup(loc: str) -> dict:
    return {"location": loc, "temperature": "72°F"}

weather_tool = Tool(
    func=_internal_weather_lookup,
    name="get_current_weather",
    description="Returns the current temperature for a given city.",
)

response = structured_completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Weather in New York?"}],
    tools=[weather_tool],
)
```

## Return Type: UnifiedResponse

All completion functions return a `UnifiedResponse` object:

| Field | Type | Description |
|-------|------|-------------|
| `content` | `BaseModel \| str \| None` | The parsed response — a Pydantic model instance if `response_model` was provided, otherwise a plain string |
| `messages` | `List[dict]` | The full conversation history including tool calls and results, ready to pass back as `messages` for multi-turn conversations |

## Error Handling

| Exception | When raised |
|-----------|-------------|
| `ModelCapabilityError` | Model doesn't support JSON output or tool calling |
| `MaxRecursionError` | Tool call chain exceeds `max_recursion` limit |
| `StructuredValidationError` | LLM output couldn't be parsed into `response_model` |
| `FunctionExecutionError` | A tool function raised an exception at runtime |

Import from `litetoolllm.errors`.

### API Reference
# structured_completion()
```python
def structured_completion(
    *,
    model: str,
    messages: List[dict],
    response_model: Optional[Type[BaseModel]] = None,
    tools: Optional[List[Callable]] = None,
    max_recursion: int = 3,
    **kwargs
) -> UnifiedResponse
```

# astructured_completion()
```python
async def astructured_completion(
    *,
    model: str,
    messages: List[dict],
    response_model: Optional[Type[BaseModel]] = None,
    tools: Optional[List[Callable]] = None,
    max_recursion: int = 3,
    **kwargs
) -> UnifiedResponse
```
