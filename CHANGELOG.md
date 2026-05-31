# Changelog

All notable changes to this project will be documented in this file.


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