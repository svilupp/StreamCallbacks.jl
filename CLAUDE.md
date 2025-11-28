# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

StreamCallbacks.jl is a Julia package that provides a unified streaming interface for Large Language Model (LLM) APIs. It handles Server-Sent Events (SSE) parsing, chunk processing, and response body reconstruction across multiple providers.

## Common Commands

```bash
# Run all tests
julia --project -e 'using Pkg; Pkg.test()'

# Run a single test file
julia --project -e 'using StreamCallbacks; include("test/stream_openai_chat.jl")'

# Start Julia REPL with project
julia --project

# Build documentation locally
julia --project=docs docs/make.jl
```

## Architecture

### Core Types (`src/interface.jl`)

- `AbstractStreamFlavor` - Base type for provider-specific streaming behavior
  - `OpenAIStream` / `OpenAIChatStream` - OpenAI Chat Completions API
  - `OpenAIResponsesStream` - OpenAI Responses API
  - `AnthropicStream`, `OllamaStream` - Anthropic and Ollama APIs
- `StreamChunk` - Represents a single SSE chunk with `event`, `data`, and parsed `json`
- `StreamCallback` - Main callback object that accumulates chunks and manages output

### Provider-Specific Files

Each provider has its own file implementing three key methods:
- `is_done(flavor, chunk)` - Detect stream termination
- `extract_content(flavor, chunk)` - Extract displayable text from chunk
- `build_response_body(flavor, cb)` - Reconstruct standard API response from chunks

| File | Provider | Flavor | Termination Signal |
|------|----------|--------|-------------------|
| `src/stream_openai_chat.jl` | OpenAI Chat Completions | `OpenAIStream` | `[DONE]` data |
| `src/stream_openai_responses.jl` | OpenAI Responses API | `OpenAIResponsesStream` | `response.completed` event |
| `src/stream_anthropic.jl` | Anthropic | `AnthropicStream` | `:message_stop` event |
| `src/stream_ollama.jl` | Ollama | `OllamaStream` | `done: true` in JSON |

### Request Flow

1. `streamed_request!(cb, url, headers, input)` - Entry point in `src/shared_methods.jl`
2. `extract_chunks(flavor, blob)` - Parse SSE blob into `StreamChunk` array (handles spillover for incomplete messages)
3. For each chunk: `callback(cb, chunk)` → `extract_content()` → `print_content()`
4. `build_response_body(flavor, cb)` - Reconstruct response mimicking non-streaming API

### Extending the Package

To add a new provider:
1. Create new `AbstractStreamFlavor` subtype
2. Implement `is_done`, `extract_content`, and `build_response_body` methods
3. Export the new flavor type

To customize output handling:
- Extend `print_content(out, text)` for custom sinks (IO, Channel, or custom types)
- Create custom callback by subtyping `AbstractStreamCallback`

## Dependencies

- HTTP.jl - HTTP requests and SSE streaming
- JSON3.jl - JSON parsing
- PrecompileTools.jl - Precompilation workloads

## Integration

This package integrates with PromptingTools.jl via `configure_callback!` which auto-configures the `StreamCallback` flavor and necessary `api_kwargs`.
