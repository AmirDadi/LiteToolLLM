# Changelog

All notable changes to this project will be documented in this file.


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