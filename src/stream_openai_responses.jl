# OpenAI Responses API Streaming
# ===============================
#
# This file implements streaming support for OpenAI's Responses API
# (POST /v1/responses with stream=true).
#
# Flavor: OpenAIResponsesStream
#
# SSE Format:
#   - Uses both `event:` and `data:` fields
#   - Event types include: response.created, response.output_text.delta,
#     response.reasoning_summary_text.delta, response.completed, etc.
#   - Content located at: delta field in delta events
#   - Stream termination: response.completed, response.failed, or response.incomplete event
#
# Example SSE message:
#   event: response.output_text.delta
#   data: {"type":"response.output_text.delta","delta":"Hello","sequence_number":4}
#
# Key differences from Chat Completions API (stream_openai_chat.jl):
#   - Richer event-based format with explicit event types
#   - Supports reasoning summaries for reasoning models
#   - No [DONE] signal - uses response.completed event instead
#   - Hierarchical structure: response → output items → content parts

"""
    is_done(flavor::OpenAIResponsesStream, chunk::AbstractStreamChunk; kwargs...)

Check if the Responses API stream is done.
Returns true when `response.completed`, `response.incomplete`, `response.failed`, or `error` event is received.
"""
@inline function is_done(flavor::OpenAIResponsesStream, chunk::AbstractStreamChunk; kwargs...)
    chunk.event in (
        Symbol("response.completed"),
        Symbol("response.incomplete"),
        Symbol("response.failed"),
        :error
    )
end

"""
    extract_content(flavor::OpenAIResponsesStream, chunk::AbstractStreamChunk;
        include_reasoning::Bool = true, kwargs...)

Extract content from Responses API chunks.
Handles both text deltas (`response.output_text.delta`) and reasoning deltas
(`response.reasoning_summary_text.delta`, `response.reasoning_text.delta`).

# Arguments
- `include_reasoning::Bool = true`: Whether to include reasoning content in output.
  Set to false to only extract final text output.
"""
@inline function extract_content(
        flavor::OpenAIResponsesStream, chunk::AbstractStreamChunk;
        include_reasoning::Bool = true, kwargs...)
    isnothing(chunk.json) && return nothing

    # Handle text output deltas (main content)
    if chunk.event == Symbol("response.output_text.delta")
        return get(chunk.json, :delta, nothing)
    end

    # Handle reasoning summary deltas (if enabled)
    if include_reasoning && chunk.event == Symbol("response.reasoning_summary_text.delta")
        return get(chunk.json, :delta, nothing)
    end

    # Handle full reasoning text deltas (if enabled)
    if include_reasoning && chunk.event == Symbol("response.reasoning_text.delta")
        return get(chunk.json, :delta, nothing)
    end

    return nothing
end

"""
    build_response_body(flavor::OpenAIResponsesStream, cb::AbstractStreamCallback;
        verbose::Bool = false, kwargs...)

Build response body from Responses API chunks to mimic a non-streaming response.
Reconstructs the final response structure with all output items, including
both text content and reasoning summaries.

The response structure follows OpenAI's Responses API format with:
- `output`: Array of output items (reasoning and message)
- `usage`: Token usage statistics (from response.completed event)
"""
function build_response_body(
        flavor::OpenAIResponsesStream, cb::AbstractStreamCallback;
        verbose::Bool = false, kwargs...)
    isempty(cb.chunks) && return nothing

    response = nothing
    text_content = IOBuffer()
    reasoning_content = IOBuffer()
    usage = nothing

    for chunk in cb.chunks
        isnothing(chunk.json) && continue

        # Capture response structure from response.completed (most complete)
        if chunk.event == Symbol("response.completed")
            resp_data = get(chunk.json, :response, nothing)
            if !isnothing(resp_data)
                response = copy(resp_data)
                # Usage is nested in the response object
                usage = get(resp_data, :usage, nothing)
            end
        end

        # Fallback: capture initial response structure from response.created
        if isnothing(response) && chunk.event == Symbol("response.created")
            resp_data = get(chunk.json, :response, nothing)
            if !isnothing(resp_data)
                response = copy(resp_data)
            end
        end

        # Accumulate text deltas
        if chunk.event == Symbol("response.output_text.delta")
            delta = get(chunk.json, :delta, nothing)
            !isnothing(delta) && write(text_content, string(delta))
        end

        # Accumulate reasoning summary deltas
        if chunk.event == Symbol("response.reasoning_summary_text.delta")
            delta = get(chunk.json, :delta, nothing)
            !isnothing(delta) && write(reasoning_content, string(delta))
        end
    end

    # Build final response structure
    if !isnothing(response)
        final_text = String(take!(text_content))
        final_reasoning = String(take!(reasoning_content))

        output = Dict{Symbol, Any}[]

        # Add reasoning output if present
        if !isempty(final_reasoning)
            push!(output, Dict{Symbol, Any}(
                :type => "reasoning",
                :summary => [Dict{Symbol, Any}(:type => "summary_text", :text => final_reasoning)]
            ))
        end

        # Add message output
        if !isempty(final_text)
            push!(output, Dict{Symbol, Any}(
                :type => "message",
                :role => "assistant",
                :content => [Dict{Symbol, Any}(:type => "output_text", :text => final_text)]
            ))
        end

        # Only override output if we assembled content
        # (response.completed already has the full output array)
        if !isempty(output) && (isempty(final_text) == false || isempty(final_reasoning) == false)
            # Check if response already has good output data from response.completed
            existing_output = get(response, :output, [])
            if isempty(existing_output)
                response[:output] = output
            end
        end

        # Ensure usage is set
        !isnothing(usage) && (response[:usage] = usage)
    end

    return response
end
