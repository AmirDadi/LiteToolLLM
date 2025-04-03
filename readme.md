# StructuralLM

**Structured Outputs & Function Calling for LLMs**  
*Transparent control flow without agent frameworks*

## Features

- **Structured Outputs**: Get validated Pydantic models from LLM responses
- **Transparent Function Calling**: Direct tool execution with full visibility
- **Model Capability Validation**: Automatic checks for JSON/tool support
- **Recursion Handling**: Configurable depth for complex tool chains
- **Sync/Async Support**: (Async coming soon!)

## Quick Start
# Basic Structured Output
```python
from structurallm import structured_completion
from pydantic import BaseModel

class Event(BaseModel):
    name: str
    year: int

response = structured_completion(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Name 3 historical events from the 1800s"}],
    response_model=list[Event]
)
for event in response.content:
    print(f"{event.name} ({event.year})")
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