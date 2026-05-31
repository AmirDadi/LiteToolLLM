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

These three tools sit at different **altitudes**, from "one typed call" to "full agent runtime" — they compose more than they compete:

- **litetoolllm** — a thin `litellm` wrapper: *one* call that may resolve tools and returns a typed Pydantic result.
- **pydantic-ai** — a type-safe single-agent framework: the agent loop, structured output, retries, and dependency injection, batteries included.
- **LangGraph** — a low-level orchestration runtime: build agents as a stateful graph of nodes/edges with cycles, persistence, and human-in-the-loop.

| | litetoolllm | pydantic-ai | LangGraph |
|---|---|---|---|
| Core abstraction | A function call | An `Agent` object | A `StateGraph` (nodes + edges) |
| Setup | ~3 lines | ~15 lines | ~30+ lines |
| Structured output | Yes (native JSON *or* `final_result` tool-output) | Yes (first-class) | Manual (validate in a node) |
| Tool calling | Yes, transparent | Yes | Yes |
| Multi-step / loops | Manual (pass `messages` back) | Built-in agent loop | Built-in, arbitrary cycles & branching |
| State / persistence | None (stateless) | Run-scoped + history | Checkpointers (durable, resumable) |
| Retries / self-correction | Not yet | Yes (`ModelRetry`) | Build as a graph edge |
| Streaming | No | Yes | Yes |
| Human-in-the-loop | No | Limited | Yes, first-class |
| Provider coverage | All litellm providers (~100+) | Growing first-class set | Many (via LangChain) |
| Overhead / weight | Minimal | Low | High |

**Choose litetoolllm** when you want one clean call that may hit a tool and hands back a typed result, across any provider — and you keep control of the flow yourself (it composes nicely *inside* a LangGraph node).
**Choose pydantic-ai** when "an agent" is the unit and you want the loop, retries, streaming, and DI without hand-rolling them.
**Choose LangGraph** when you need a multi-step, branching, resumable workflow with durable state and human approval steps.

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
| OpenAI (gpt-4o, gpt-4o-mini, etc.) | Yes | Yes | Native JSON mode |
| Google Gemini | Yes | Yes | Native JSON mode |
| Anthropic Claude | Yes | Yes | Via `final_result` tool-output mode |
| Mistral | Yes | Yes | |
| Groq | Yes | Yes | Via `final_result` tool-output mode |

litetoolllm picks the path automatically: **native JSON mode** when the model
supports JSON-schema output, otherwise a **result-function (tool-output) mode**
that delivers the structured answer through a synthetic `final_result` tool. So
any model with either JSON-schema *or* tool-calling support can return structured
output.

Capability detection is delegated to litellm (`litellm.supports_response_schema`
/ `litellm.supports_function_calling`). When that detection is wrong (common via
OpenRouter), pass `model_capabilities` to override it:

```python
structured_completion(
    model="openrouter/meta-llama/llama-3.3-70b-instruct",
    messages=[...],
    response_model=MyModel,
    model_capabilities={"function_calling": True, "json_mode": False},
)
```

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
| `usage` | `Usage \| None` | Token usage from the final model response (best-effort; `None` if unavailable) |
| `cost` | `float \| None` | Estimated dollar cost of the final response (best-effort; `None` for unknown/custom models) |

## Error Handling

Every exception inherits from `LiteToolLLMError`, so you can catch them all in one place:

```python
from litetoolllm.errors import LiteToolLLMError

try:
    structured_completion(...)
except LiteToolLLMError as e:
    ...  # any litetoolllm error
```

| Exception | When raised |
|-----------|-------------|
| `LiteToolLLMError` | Base class for all errors below |
| `ModelCapabilityError` | Model doesn't support JSON output or tool calling |
| `MaxRecursionError` | Tool call chain exceeds `max_recursion` limit |
| `StructuredValidationError` | LLM output couldn't be parsed into `response_model` |
| `FunctionExecutionError` | A tool function raised at runtime (exposes `.function_name`, `.details`, `.tool_call`; original error in `__cause__`) |

Import from `litetoolllm.errors`.

## Observability & Tracing

Each `UnifiedResponse` carries the final response's `usage` and `cost` for lightweight tracking:

```python
result = structured_completion(model="gpt-4o", messages=messages, response_model=MyModel)
print(result.usage)  # token counts
print(result.cost)   # estimated $ cost
```

For full tracing, litetoolllm calls LiteLLM directly, so LiteLLM's built-in
integrations (Langfuse, OpenTelemetry, etc.) work without any wrapper-specific
setup. Register the callback globally once:

```python
import litellm
litellm.success_callback = ["langfuse"]   # set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY env vars
```

Anything you pass as `metadata=` is forwarded straight to LiteLLM, which Langfuse
uses for trace correlation (`trace_id`, `session_id`, `tags`, …):

```python
structured_completion(
    model="gpt-4o",
    messages=messages,
    response_model=MyModel,
    metadata={"trace_id": "abc-123", "session_id": "user-42", "tags": ["prod"]},
)
```

See the [LiteLLM logging docs](https://docs.litellm.ai/docs/observability/langfuse_integration) for other backends.

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
    metadata: Optional[dict] = None,
    model_capabilities: Optional[dict] = None,
    **kwargs
) -> UnifiedResponse
```

`model_capabilities` (optional): `{"function_calling": bool, "json_mode": bool}`
to override litellm's auto-detection for that call. Omitted keys fall back to
detection.

# astructured_completion()
```python
async def astructured_completion(
    *,
    model: str,
    messages: List[dict],
    response_model: Optional[Type[BaseModel]] = None,
    tools: Optional[List[Callable]] = None,
    max_recursion: int = 3,
    metadata: Optional[dict] = None,
    model_capabilities: Optional[dict] = None,
    **kwargs
) -> UnifiedResponse
```
