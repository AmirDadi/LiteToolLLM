# Result-Function (Tool-Output) Structured Output — Design

**Date:** 2026-05-31
**Status:** Approved (design); pending implementation plan
**Branch:** `feat/result-function-output-mode`

## Problem

Today `litetoolllm` produces structured output by passing `response_format=response_model`
to LiteLLM (native JSON / JSON-schema mode) and validating the JSON content. This has
limits:

- **Provider coverage is narrow.** Native JSON-schema mode is unavailable on many
  models and unreliable through gateways like OpenRouter (e.g. `litellm.supports_response_schema`
  returns `False` for `openrouter/openai/gpt-4o`).
- **Tools + structured output don't mix** on some providers. Gemini cannot do tool
  calling and JSON output in one call, so the async path carries a special-case that
  makes a *second* model call purely to reformat — extra latency and a maintenance wart.
- **Capability detection is sometimes wrong.** `litellm.supports_function_calling`
  returns `False` for `openrouter/meta-llama/llama-3.3-70b-instruct`, yet tool calling
  works against the real endpoint (verified). Trusting the flag blindly would reject
  models that actually work.

## Goals

1. Remove the Gemini manual (two-call) handling.
2. Support Anthropic and Groq-class (Llama) models.
3. Support **any** LLM with tool/function-calling support via a result-function path.
4. Support **any** LLM without tool support but with JSON output via the existing path.
5. **Backward compatible**: all current tests must pass unchanged; public signatures only
   gain optional keyword arguments.

Out of scope: retry / self-correction (`ModelRetry`-style loops). Tracked separately.

## Approach

Introduce a **result-function (tool-output) mode**: register the `response_model` as a
synthetic `final_result` tool whose parameters are the model's JSON schema. The model
delivers the structured answer by "calling" that tool over the normal function-calling
channel; we validate its arguments. JSON mode is retained for models that lack tool
support. A pure routing function selects the path per call.

## Routing

Capabilities resolve from an optional caller override merged over LiteLLM detection:

- `caps.function_calling = model_capabilities["function_calling"]` if provided,
  else `litellm.supports_function_calling(model)`.
- `caps.json_mode = model_capabilities["json_mode"]` if provided,
  else `litellm.supports_response_schema(model)`.

**Principle: JSON mode wins whenever it is available; tool-output only fills the gap**
for models that can call tools but lack native JSON-schema support. This keeps every
existing OpenAI/Gemini behavior (and its tests) byte-for-byte unchanged, and verified
testing showed LiteLLM now handles Gemini `tools + response_format` natively — so the
old Gemini two-call hack is simply deleted, with Gemini routed through the standard JSON
path.

`select_output_mode(response_model, tools, caps)` implements this table:

| `response_model` | user `tools` | `function_calling` | `json_mode` | → Mode |
|---|---|---|---|---|
| yes | any | (if tools) no | —   | raise `ModelCapabilityError` (can't run the tools) |
| yes | any | —   | yes | **JSON** (current path, unchanged) — OpenAI, Gemini |
| yes | any | yes | no  | **TOOL_OUTPUT** (`final_result` + any user tools) — Anthropic, Llama/Groq |
| yes | any | no  | no  | raise `ModelCapabilityError` |
| no  | yes | yes | —   | **NONE** — plain tool loop (current path) |
| no  | yes | no  | —   | raise `ModelCapabilityError` |
| no  | no  | —   | —   | **NONE** — plain completion (current path) |

Evaluation order (pure function): (1) no `response_model` and no `tools` → `NONE`;
(2) `tools` present and `function_calling` is False → raise (can't run the tools);
(3) no `response_model` → `NONE` (plain tool loop); (4) `json_mode` → `JSON`;
(5) `function_calling` → `TOOL_OUTPUT`; (6) otherwise raise `ModelCapabilityError`.

Consequences:
- JSON-capable models (OpenAI, Gemini) always use native JSON mode, with or without
  tools — identical to today, so existing tests pass unchanged.
- The Gemini `tools + response_model` two-call hack is deleted; Gemini uses the standard
  JSON path (verified working end-to-end).
- Tool-output is exercised only by models with tools but no JSON-schema support
  (Anthropic, Llama/Groq via OpenRouter), satisfying "any LLM with tool support."

## Module structure

Keep `core.py` as a thin orchestrator; put the decision logic and path mechanics in small,
pure, independently testable units.

### New: `litetoolllm/output_modes.py`
- `Capabilities` — small dataclass: `function_calling: bool`, `json_mode: bool`.
- `resolve_capabilities(model, model_capabilities) -> Capabilities` — merge override over
  LiteLLM detection. Single source of truth.
- `OutputMode` — enum: `JSON`, `TOOL_OUTPUT`, `NONE`.
- `select_output_mode(response_model, tools, caps) -> OutputMode` — pure function
  implementing the routing table; raises `ModelCapabilityError` for unsupported rows.
- `FINAL_RESULT_TOOL_NAME = "final_result"`.
- `build_final_result_tool(response_model) -> dict` — `response_model.model_json_schema()`
  wrapped as a `{"type": "function", "function": {...}}` tool dict.
- `parse_final_result(arguments, response_model) -> BaseModel` —
  `response_model.model_validate(arguments)` (arguments is a dict from the tool call).

### Changed: `litetoolllm/utils.py`
- Helper to find a `final_result` tool call within a response's `tool_calls`.
- Ensure `final_result` is excluded from the user function-mapping so it is never executed
  as a real function.

### Changed: `litetoolllm/core.py`
- Compute `caps = resolve_capabilities(...)` and `mode = select_output_mode(...)` once.
- Dispatch to private helpers:
  - `_run_json_mode(...)` — today's logic, essentially unchanged.
  - `_run_tool_output_mode(...)` — new (see Data flow).
- Sync and async mirror each other. The Gemini block is removed.

## Data flow — TOOL_OUTPUT mode

1. `tools_payload = convert_tools_to_api_format(user_tools) + [build_final_result_tool(response_model)]`.
   Do **not** pass `response_format`.
2. Call the model with `tool_choice="auto"`.
3. In the existing recursion loop, for each returned tool call:
   - **user tool** → execute as today, append the tool result message, continue;
   - **`final_result`** → `parse_final_result(args, response_model)`, set as content,
     **stop** (do not recurse, do not execute it as a function).
4. If the loop ends (model returned content with no tool calls) and `final_result` was
   never seen → **one forced extraction call**: resend the conversation with `tool_choice`
   forcing `final_result`, then parse that call's arguments.
5. `max_recursion` bounds the user-tool loop; the forced extraction is a single extra step
   outside it.

## Error handling

Reuses the v0.1.11 hierarchy; no new public exceptions.

- `final_result` arguments fail validation → `StructuredValidationError`
  (with `retry_context` = raw response), matching JSON mode today.
- Forced extraction call still yields no `final_result` → `StructuredValidationError`.
- TOOL_OUTPUT required but `function_calling` is `False` (and no override) →
  `ModelCapabilityError` (preserves the current test).
- User-tool runtime failures → `FunctionExecutionError`, unchanged.

## Public API changes

`structured_completion` / `astructured_completion` gain one keyword-only argument:

```python
def structured_completion(*, model, messages, response_model=None, tools=None,
                          max_recursion=3, metadata=None,
                          model_capabilities=None, **kwargs) -> UnifiedResponse: ...
```

- `model_capabilities: Optional[dict]` — e.g. `{"function_calling": True, "json_mode": False}`.
  When provided, bypasses LiteLLM auto-detection for that call. Default `None` → current
  behavior. `UnifiedResponse` is unchanged.

## Testing

### Deterministic (no API, always run)
- `select_output_mode` — every routing-table row, including the `ModelCapabilityError`
  rows.
- `resolve_capabilities` — override precedence over LiteLLM detection.
- `build_final_result_tool` — schema shape from a sample model.
- `parse_final_result` — happy path and invalid-arguments path.
- `final_result` detection in a response and exclusion from user function execution.

### Live (OpenRouter; auto-skip unless `OPENROUTER_API_KEY` is set)
- Structured output via TOOL_OUTPUT on `openrouter/anthropic/claude-sonnet-4.6`.
- Structured output via TOOL_OUTPUT on `openrouter/meta-llama/llama-3.3-70b-instruct`
  using `model_capabilities={"function_calling": True}` (litellm misreports this model).
- User tools + `response_model` together (the path that replaces the Gemini hack).
- Prose-fallback path (model answers without calling `final_result` → forced extraction).
- Gemini tools + `response_model` live test to prove the deleted special-case is covered
  by the new path.

### Backward-compat
- Existing OpenAI/Gemini suite runs unchanged and must stay green.

## Backward compatibility & risks

- Only additive, keyword-only API change; default path equals today's behavior.
- Risk: forced-extraction adds at most one extra call, only in the prose-fallback case.
- Risk: provider differences in honoring `tool_choice` forcing — covered by live tests.
- Versioning: backward-compatible feature → minor bump (proposed `0.2.0`).
