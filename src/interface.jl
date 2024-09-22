# This file defines the core interface for the StreamCallbacks.jl package
#
# The goal is to enable custom callbacks for streaming LLM APIs, 
#   but also re-construct the response body in a standard way in the end 
#   to mimic non-streaming responses for downstream applications
#
# The below functions can be extended with custom methods to achieve the desired behavior

# ## Interface
# It all revolves around the `StreamCallback` object, 
# which is a simple struct that holds the individual "chunks" (StreamChunk)
# and presents the logic necessary for processing
#
# Top-level interface that wraps the HTTP.POST request and handles the streaming
function streamed_request! end
# It composes of the following interface functions
# Extract the chunks from the received SSE blob. Returns a list of `StreamChunk`
# At the moment, it's assumed to be generic enough for ANY API provider (TBU).
function extract_chunks end
# This is a utility to check if the stream of message is done for a given API provider -- "flavor"
function is_done end
# Main function for user logic - it's provided each "chunk" and user decides what to do with it
function callback end
# This is a utility INSIDE the callback function, 
# to extract the content from the chunk for a given API provider -- "flavor"
function extract_content end
# This ia a utility INSIDE to callback function to support different "sinks" to stream the content into
function print_content end
# This is a utility to build the response body from the chunks
# to mimic receiving a standard response from the API -- depends on the API provider "flavor"
function build_response_body end

# ## Types
"""
    AbstractStreamChunk

Abstract type for the stream chunk.

It must have the following fields:
- `event`: The event name.
- `data`: The data chunk.
- `json`: The JSON object or `nothing` if the chunk does not contain JSON.
"""
abstract type AbstractStreamChunk end

"""
    AbstractStreamCallback

Abstract type for the stream callback.

It must have the following fields:
- `out`: The output stream, eg, `stdout` or a pipe.
- `flavor`: The stream flavor which might or might not differ between different providers, eg, `OpenAIStream` or `AnthropicStream`.
- `chunks`: The list of received `AbstractStreamChunk` chunks.
- `verbose`: Whether to print verbose information.
- `throw_on_error`: Whether to throw an error if an error message is detected in the streaming response.
- `kwargs`: Any custom keyword arguments required for your use case.
"""
abstract type AbstractStreamCallback end

"""
    AbstractStreamFlavor

Abstract type for the stream flavor, ie, the API provider.

Available flavors:
- `OpenAIStream` for OpenAI API
- `AnthropicStream` for Anthropic API
"""
abstract type AbstractStreamFlavor end
struct OpenAIStream <: AbstractStreamFlavor end
struct AnthropicStream <: AbstractStreamFlavor end

## Default implementations
"""
    StreamChunk

A chunk of streaming data. A message is composed of multiple chunks.

# Fields
- `event`: The event name.
- `data`: The data chunk.
- `json`: The JSON object or `nothing` if the chunk does not contain JSON.
"""
@kwdef struct StreamChunk{T1 <: AbstractString, T2 <: Union{JSON3.Object, Nothing}} <:
              AbstractStreamChunk
    event::Union{Symbol, Nothing} = nothing
    data::T1 = ""
    json::T2 = nothing
end
function Base.show(io::IO, chunk::StreamChunk)
    data_preview = if length(chunk.data) > 10
        "$(first(chunk.data, 10))..."
    else
        chunk.data
    end
    json_keys = if !isnothing(chunk.json)
        join(keys(chunk.json), ", ", " and ")
    else
        "-"
    end
    print(io,
        "StreamChunk(event=$(chunk.event), data=$(data_preview), json keys=$(json_keys))")
end

"""
    StreamCallback

Simplest callback for streaming message, which just prints the content to the output stream defined by `out`.
When streaming is over, it builds the response body from the chunks and returns it as if it was a normal response from the API.

For more complex use cases, you can define your own `callback`. See the interface description below for more information.

# Fields
- `out`: The output stream, eg, `stdout` or a pipe.
- `flavor`: The stream flavor which might or might not differ between different providers, eg, `OpenAIStream` or `AnthropicStream`.
- `chunks`: The list of received `StreamChunk` chunks.
- `verbose`: Whether to print verbose information. If you enable DEBUG logging, you will see the chunks as they come in.
- `throw_on_error`: Whether to throw an error if an error message is detected in the streaming response.
- `kwargs`: Any custom keyword arguments required for your use case.

# Interface

- `StreamCallback(; kwargs...)`: Constructor for the `StreamCallback` object.
- `streamed_request!(cb, url, headers, input)`: End-to-end wrapper for POST streaming requests.

`streamed_request!` composes of:
- `extract_chunks(flavor, blob)`: Extract the chunks from the received SSE blob. Returns a list of `StreamChunk` and the next spillover (if message was incomplete).
- `callback(cb, chunk)`: Process the chunk to be printed
    - `extract_content(flavor, chunk)`: Extract the content from the chunk.
    - `print_content(out, text)`: Print the content to the output stream.
- `is_done(flavor, chunk)`: Check if the stream is done.
- `build_response_body(flavor, cb)`: Build the response body from the chunks to mimic receiving a standard response from the API.

If you want to implement your own callback, you can create your own methods for the interface functions.
Eg, if you want to print the streamed chunks into some specialized sink or Channel, you could define a simple method just for `print_content`.

# Example
```julia
using PromptingTools
const PT = PromptingTools

# Simplest usage, just provide where to steam the text (we build the callback for you)
msg = aigenerate("Count from 1 to 100."; streamcallback = stdout)

streamcallback = PT.StreamCallback() # record all chunks
msg = aigenerate("Count from 1 to 100."; streamcallback)
# this allows you to inspect each chunk with `streamcallback.chunks`

# Get verbose output with details of each chunk for debugging
streamcallback = PT.StreamCallback(; verbose=true, throw_on_error=true)
msg = aigenerate("Count from 1 to 10."; streamcallback)
```

Note: If you provide a `StreamCallback` object to `aigenerate`, we will configure it and necessary `api_kwargs` via `configure_callback!` unless you specify the `flavor` field.
If you provide a `StreamCallback` with a specific `flavor`, we leave all configuration to the user (eg, you need to provide the correct `api_kwargs`).
"""
@kwdef mutable struct StreamCallback{T1 <: Any} <: AbstractStreamCallback
    out::T1 = stdout
    flavor::Union{AbstractStreamFlavor, Nothing} = nothing
    chunks::Vector{<:StreamChunk} = StreamChunk[]
    verbose::Bool = false
    throw_on_error::Bool = false
    kwargs::NamedTuple = NamedTuple()
end
function Base.show(io::IO, cb::StreamCallback)
    print(io,
        "StreamCallback(out=$(cb.out), flavor=$(cb.flavor), chunks=$(length(cb.chunks)) items, $(cb.verbose ? "verbose" : "silent"), $(cb.throw_on_error ? "throw_on_error" : "no_throw"))")
end
Base.empty!(cb::AbstractStreamCallback) = empty!(cb.chunks)
Base.push!(cb::AbstractStreamCallback, chunk::StreamChunk) = push!(cb.chunks, chunk)
Base.isempty(cb::AbstractStreamCallback) = isempty(cb.chunks)
Base.length(cb::AbstractStreamCallback) = length(cb.chunks)
