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

## Installation

You can install litetoolllm directly from the GitHub repository:

```bash
pip install git+https://github.com/AmirDadi/liteToolLlm.git
```

For development installation with additional testing dependencies:

```bash
pip install git+https://github.com/AmirDadi/liteToolLlm.git#egg=litetoolllm[dev]
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