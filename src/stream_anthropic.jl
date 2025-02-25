# Custom methods for Anthropic streaming -- flavor=AnthropicStream()

@inline function is_done(flavor::AnthropicStream, chunk::AbstractStreamChunk; kwargs...)
    chunk.event == :error || chunk.event == :message_stop
end

"""
    extract_content(flavor::AnthropicStream, chunk::AbstractStreamChunk; kwargs...)

Extract the content from the chunk.
"""
function extract_content(flavor::AnthropicStream, chunk::AbstractStreamChunk; kwargs...)
    isnothing(chunk.json) && return nothing
    
    # Handle text_delta format specifically
    if get(chunk.json, :type, nothing) == "content_block_delta" &&
       haskey(chunk.json, :delta) &&
       get(chunk.json[:delta], :type, nothing) == "text_delta"
        return get(chunk.json[:delta], :text, nothing)
    end
    
    # Handle other formats (original implementation)
    index = get(chunk.json, :index, nothing)
    isnothing(index) || !iszero(index) && return nothing

    delta_block = get(chunk.json, :content_block, nothing)
    if isnothing(delta_block)
        # look for the delta segment
        delta_block = get(chunk.json, :delta, Dict())
    end
    
    get(delta_block, :text, nothing)
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
            response = isnothing(response) ? get(chunk.json, :delta, Dict()) :
                       merge(response, get(chunk.json, :delta, Dict()))
            usage = isnothing(usage) ? get(chunk.json, :usage, Dict()) :
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
        response[:content] = [Dict(:type => "text", :text => String(take!(content_buf)))]
        !isnothing(usage) && (response[:usage] = usage)
    end
    return response
end
