# Custom methods for Anthropic streaming -- flavor=AnthropicStream()

@inline function is_done(flavor::AnthropicStream, chunk::AbstractStreamChunk; kwargs...)
    chunk.event == :error || chunk.event == :message_stop
end

"""
    extract_content(flavor::AnthropicStream, chunk::AbstractStreamChunk;
        include_thinking::Bool = true, kwargs...)

Extract the content from the chunk.
"""
function extract_content(flavor::AnthropicStream, chunk::AbstractStreamChunk;
        include_thinking::Bool = true, kwargs...)
    isnothing(chunk.json) && return nothing

    # Get chunk type
    chunk_type = get(chunk.json, :type, nothing)

    # Handle content blocks (start and stop)
    if chunk_type == "content_block_start" || chunk_type == "content_block_stop"
        content_block = get(chunk.json, :content_block, Dict())
        block_type = get(content_block, :type, nothing)

        # Extract text content
        if block_type == "text"
            return get(content_block, :text, nothing)
            # Extract thinking content if enabled
        elseif include_thinking && block_type == "thinking"
            return get(content_block, :thinking, nothing)
        end
        # Handle content block deltas
    elseif chunk_type == "content_block_delta"
        delta = get(chunk.json, :delta, Dict())
        delta_type = get(delta, :type, nothing)

        # Extract text delta content
        if delta_type == "text_delta"
            return get(delta, :text, nothing)
            # Extract thinking delta content if enabled
        elseif include_thinking && delta_type == "thinking_delta"
            return get(delta, :thinking, nothing)
        end
    end

    return nothing
end

"""
    build_response_body(
        flavor::AnthropicStream, cb::AbstractStreamCallback; verbose::Bool = false, kwargs...)

Build the response body from the chunks to mimic receiving a standard response from the API.

Note: Limited functionality for now. Does NOT support tool use. Use standard responses for these.
"""
function build_response_body(
        flavor::AnthropicStream, cb::AbstractStreamCallback; verbose::Bool = false, kwargs...)
    isempty(cb.chunks) && return nothing
    response = nothing
    usage = nothing
    content_buf = IOBuffer()
    for i in eachindex(cb.chunks)
        ## Note we ignore the index ID, because Anthropic does not support multiple
        ## parallel generations
        chunk = cb.chunks[i]
        ## validate that we can access choices
        isnothing(chunk.json) && continue
        ## Core of the message body
        if isnothing(response) && chunk.event == :message_start &&
           haskey(chunk.json, :message)
            ## do it only once the first time when we have the json
            response = chunk.json[:message] |> copy
            usage = get(response, :usage, Dict())
        end
        ## Update stop reason and usage
        if chunk.event == :message_delta
            response = isnothing(response) ? get(copy(chunk.json), :delta, Dict()) :
                       merge(response, get(chunk.json, :delta, Dict()))
            usage = isnothing(usage) ? get(copy(chunk.json), :usage, Dict()) :
                    merge(usage, get(chunk.json, :usage, Dict()))
        end

        ## Load text chunks
        if chunk.event == :content_block_start ||
           chunk.event == :content_block_delta || chunk.event == :content_block_stop
            ## Find the text delta
            delta_block = get(chunk.json, :content_block, nothing)
            if isnothing(delta_block)
                ## look for the delta segment
                delta_block = get(chunk.json, :delta, Dict())
            end
            text = get(delta_block, :text, nothing)
            !isnothing(text) && write(content_buf, text)
        end
    end
    ## We know we have at least one chunk, let's use it for final response
    if !isnothing(response)
        ## We need to make sure we're operating on a copy of the response
        response isa JSON3.Object && (response = copy(response))
        response[:content] = [Dict(:type => "text", :text => String(take!(content_buf)))]
        !isnothing(usage) && (response[:usage] = usage)
    end
    return response
end
