# StreamCallbacks.jl 
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://svilupp.github.io/StreamCallbacks.jl/stable/) 
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://svilupp.github.io/StreamCallbacks.jl/dev/) 
[![Build Status](https://github.com/svilupp/StreamCallbacks.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/svilupp/StreamCallbacks.jl/actions/workflows/CI.yml?query=branch%3Amain) 
[![Coverage](https://codecov.io/gh/svilupp/StreamCallbacks.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/svilupp/StreamCallbacks.jl) 
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

StreamCallbacks.jl is designed to unify streaming interfaces for Large Language Models (LLMs) across multiple providers. It simplifies handling Server-Sent Events (SSE), provides easy debugging by collecting all chunks, and offers various built-in sinks (e.g., `stdout`, channels, pipes) for streaming data. You can also extend it to implement custom logic for processing streamed data.

## Table of Contents

- [StreamCallbacks.jl](#streamcallbacksjl)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Supported Providers](#supported-providers)
  - [Installation](#installation)
  - [Getting Started](#getting-started)
  - [Usage Examples](#usage-examples)
    - [Example with OpenAI API](#example-with-openai-api)
    - [Example with PromptingTools.jl](#example-with-promptingtoolsjl)
  - [Extending StreamCallbacks.jl](#extending-streamcallbacksjl)
    - [StreamCallback Interface](#streamcallback-interface)
    - [Custom Callback Example](#custom-callback-example)

## Features

- **Unified Streaming Interface**: Provides a consistent API for streaming responses from various LLM providers.
- **Easy Debugging**: Collects all received chunks, enabling detailed inspection and debugging.
- **Built-in Sinks**: Supports common sinks like `stdout`, channels, and pipes out of the box.
- **Customizable Callbacks**: Extendable interface allows you to define custom behavior for each received chunk.

## Supported Providers

- **OpenAI API** (and all compatible providers)
- **Anthropic API**
- **Ollama API** (`api/chat` endpoint, OpenAI-compatible endpoint)

## Installation

You can install StreamCallbacks.jl via the package manager:

```julia
import Pkg
Pkg.add("StreamCallbacks")
```

## Getting Started

StreamCallbacks.jl revolves around the `StreamCallback` type, which manages the streaming of messages and the handling of received chunks. Here's a simple example of how to use it:

```julia
using StreamCallbacks

# Create a StreamCallback object that streams output to stdout
cb = StreamCallback(out = stdout)

# Use the callback with your API request (see Usage Examples below)
```

## Usage Examples

### Example with OpenAI API

```julia
using HTTP
using JSON3
using StreamCallbacks

# Prepare target URL and headers
url = "https://api.openai.com/v1/chat/completions"
headers = [
    "Content-Type" => "application/json",
    "Authorization" => "Bearer $(get(ENV, "OPENAI_API_KEY", ""))"
]

# Create a StreamCallback object
cb = StreamCallback(out = stdout, flavor = OpenAIStream())

# Prepare the request payload
messages = [Dict("role" => "user", "content" => "Count from 1 to 100.")]
payload = IOBuffer()
JSON3.write(payload, (; stream = true, messages, model = "gpt-4o-mini", stream_options = (; include_usage = true)))

# Send the streamed request
resp = streamed_request!(cb, url, headers, payload)

# Check the response
println("Response status: ", resp.status)
```

**Note**: For debugging, you can set `verbose = true` in the `StreamCallback` constructor to get detailed logs of each chunk. Ensure you enable DEBUG logging level in your environment.

### Example with PromptingTools.jl

StreamCallbacks.jl is integrated with [PromptingTools.jl](https://github.com/JuliaAI/PromptingTools.jl), allowing you to easily handle streaming in AI generation tasks.

```julia
using PromptingTools
const PT = PromptingTools

# Simplest usage: stream output to stdout (the callback is built for you)
msg = aigenerate("Count from 1 to 100."; streamcallback = stdout)

# Create a StreamCallback object to record all chunks
streamcallback = PT.StreamCallback()
msg = aigenerate("Count from 1 to 100."; streamcallback)
# You can inspect each chunk with `streamcallback.chunks`

# Get verbose output with details of each chunk for debugging
streamcallback = PT.StreamCallback(verbose = true, throw_on_error = true)
msg = aigenerate("Count from 1 to 10."; streamcallback)
```

**Note**: If you provide a `StreamCallback` object to `aigenerate`, PromptingTools.jl will configure it and necessary `api_kwargs` via `configure_callback!` unless you specify the `flavor` field. If you provide a `StreamCallback` with a specific `flavor`, you need to provide the correct `api_kwargs` yourself.

## Extending StreamCallbacks.jl

For more complex use cases, you can define your own `callback` methods. This allows you to customize how each chunk is processed. Here's how the interface works:

### StreamCallback Interface

- **Constructor**: `StreamCallback(; kwargs...)` creates a new `StreamCallback` object.
- **`streamed_request!`**: `streamed_request!(cb, url, headers, input)` sends a streaming POST request and processes the response using the callback.

The `streamed_request!` function internally calls:

- **`extract_chunks`**: `extract_chunks(flavor, blob)` extracts chunks from the received SSE blob.
- **`callback`**: `callback(cb, chunk)` processes each received chunk.
    - **`extract_content`**: `extract_content(flavor, chunk)` extracts the content from the chunk.
    - **`print_content`**: `print_content(out, text)` prints the content to the output stream.
- **`is_done`**: `is_done(flavor, chunk)` checks if the streaming is complete.
- **`build_response_body`**: `build_response_body(flavor, cb)` builds the final response body from the collected chunks.

### Custom Callback Example

Suppose you want to process each chunk and send it to a custom sink, such as a logging system or a GUI component. You can extend the `print_content` method:

```julia
using StreamCallbacks

struct MyCustomCallback <: StreamCallbacks.AbstractCallback
    out::IO
    # ... add additional fields if necessary
end

function StreamCallbacks.callback(cb::MyCustomCallback, chunk::StreamChunk; kwargs...)
    # Custom logic to handle the text
    println("Received chunk: ", chunk.data)
    # For example, send the text to a GUI component or log it
end
```