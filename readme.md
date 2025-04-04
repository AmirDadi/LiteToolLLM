
# LiteCallLLM

LiteCallLLM is a lightweight wrapper built on top of [litellm](https://github.com/litellm/litellm) that seamlessly integrates structured output validation with transparent tool calling for large language models. While litellm already provides powerful features such as synchronous (`completion`) and asynchronous (`acompletion`) completions, JSON-based response schemas, and built-in function calling capabilities, LiteCallLLM extends these functionalities with a clear, debuggable mechanism for directly invoking tools based on LLM outputs—without the complexity of traditional agent frameworks.

**Key Features:**

- **Transparent Tool Calling:**  
  Easily trace and debug function invocations directly from your LLM responses, giving you full visibility into which tools are executed and when.

- **Structured Output Validation:**  
  Use Pydantic models to ensure that responses conform to your expected schema, enhancing reliability and simplifying downstream processing.

- **Automated Tool Call Handling:**  
  No need to manually handle tool call responses or convert outputs. LiteCallLLM automatically processes tool responses, streamlining your workflow.

- **Full litellm Integration:**  
  Leverage all the features of litellm—including synchronous and asynchronous completions, JSON mode support, and function calling—while gaining additional control and transparency.

- **Minimalistic and Intuitive:**  
  Designed to avoid the overhead and complexity of frameworks like LangChain or LangGraph, LiteCallLLM offers a straightforward, powerful solution for advanced LLM interactions.
- **Recursion Handling**: 
  Configurable depth for complex tool chains
- **Sync/Async Support**:
  (Async coming soon!)

## Usage Examples

### 1. Single Tool Execution

Invoke a tool (e.g., a weather lookup) and have the response validated against a structured schema.

```python
response = structured_completion(
    model="gemini/gemini-2.0-flash",
    messages=[{"role": "user", "content": "What is the weather in San Francisco?"}],
    response_model=Temperature,
    tools=[get_current_weather],
    max_recursion=10
)

```
### 2. Single Tool Execution without Schema
Use LiteCallLLM without providing a response model. The output will be returned as a plain string.

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