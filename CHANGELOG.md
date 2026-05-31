# Changelog

All notable changes to this project will be documented in this file.


## [0.1.11] - 2026-05-31

### Added
- `LiteToolLLMError` base exception; every library error now inherits from it, so callers can `except LiteToolLLMError` to catch any error in one place
- `FunctionExecutionError.tool_call` attribute, exposing the raw model tool-call alongside `function_name` and `details`
- `UnifiedResponse.usage` and `UnifiedResponse.cost` fields, populated (best-effort) from the final model response for token/cost observability
- `get_usage_and_cost()` helper in `utils`
- README: "Observability & Tracing" section documenting the `usage`/`cost` fields and the LiteLLM `success_callback` + `metadata` pattern for Langfuse/OpenTelemetry tracing

### Changed
- `FunctionExecutionError.function_name` now holds the actual tool name string (previously it received the raw tool-call object)
- Async tool failures now raise `FunctionExecutionError` (consistent with the sync path) instead of letting the raw exception propagate; the original error is preserved as `__cause__`
- Added debug logging to the async tool-call path (previously only the sync path logged) and a warning log when response validation fails
- `RecursionDepthExceedError` is deprecated in favour of `MaxRecursionError` (kept as a subclass of `LiteToolLLMError` for backward compatibility)
- Async tool execution computes the function mapping once per turn instead of once per tool call

## [0.1.10] - 2026-05-31

### Added
- MIT `LICENSE` file (Copyright Amirreza Dadfarnia (AIR))
- README: "When to Use litetoolllm" positioning section vs pydantic-ai and LangChain
- README: model compatibility matrix
- README: `Tool` class usage example
- README: `UnifiedResponse` return-type documentation
- README: error-handling section documenting the exception hierarchy

## [0.1.9] - 2026-05-31

### Fixed
- Pydantic v2 compatibility: replaced removed `parse_raw()` with `model_validate_json()` in both sync and async paths
- Tool calling no longer forces every tool to accept a `metadata` parameter; `metadata` is now injected only when the function signature accepts it, and any `metadata` echoed back by the model is stripped first

### Changed
- Replaced the debug `print()` in tool-call handling with a standard `logging` call (`logging.getLogger(__name__)`)

### Added
- Deterministic (no-API) unit tests for metadata handling, plus integration tests for metadata-free tools and the `Tool` wrapper
- `.gitignore` for Python build/test artifacts; stopped tracking `__pycache__`

## [0.1.1] 2025-04-05

### Added
- Async completion support with `astructured_completion` function
- Parallel tool execution in async mode
- Comprehensive async test suite
- pytest-asyncio integration for async testing

### Changed
- Reorganized code structure:
  - Moved utility functions to `utils.py`
  - Kept main API functions in `core.py`
  - Improved code modularity and maintainability
- Updated README with async usage examples
- Added proper async error handling

### Fixed
- Fixed async tool execution implementation
- Improved error handling in async functions
- Fixed parallel tool execution in async mode

## [0.1.0] - 2025-04-04

### Added
- Initial release
- Basic structured completion functionality
- Tool calling support
- Response model validation
- Recursion handling for tool chains 