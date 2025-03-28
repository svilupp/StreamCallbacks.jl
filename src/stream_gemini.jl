# This file defines the methods for the GoogleStream flavor

"""
    extract_chunks(::GoogleStream, blob::AbstractString; kwargs...)

Extract the chunks from the received SSE blob for Google Gemini API.
Returns a list of `StreamChunk` and the next spillover (if message was incomplete).
"""
function extract_chunks(::GoogleStream, blob::AbstractString;
        spillover::AbstractString = "", verbose::Bool = false, kwargs...)
    chunks = StreamChunk[]
    next_spillover = ""
    
    # Gemini uses simpler SSE format with just "data:" prefix
    # Split by double newlines which separate SSE events
    blob_split = split(blob, r"\r?\n\r?\n")
    
    for chunk_data in blob_split
        isempty(chunk_data) && continue
        
        # Extract data after "data:" prefix
        if startswith(chunk_data, "data: ")
            json_str = chunk_data[7:end]  # Skip "" prefix
            
            # Try to parse JSON
            json_obj = nothing
            try
                json_obj = JSON3.read(json_str)
            catch e
                verbose && @warn "Cannot parse Gemini JSON: $json_str" exception=e
            end
            
            # Create chunk
            push!(chunks, StreamChunk(nothing, json_str, json_obj))
        end
    end
    
    return chunks, next_spillover
end

"""
    is_done(::GoogleStream, chunk::AbstractStreamChunk; kwargs...)

Check if the stream is done for Google Gemini API.
"""
function is_done(::GoogleStream, chunk::AbstractStreamChunk; kwargs...)
    if !isnothing(chunk.json)
        # Check for completion markers in Gemini response
        if haskey(chunk.json, :candidates) && length(chunk.json.candidates) > 0
            candidate = chunk.json.candidates[1]
            return haskey(candidate, :finishReason) && candidate.finishReason == "STOP"
        end
    end
    return false
end

"""
    extract_content(::GoogleStream, chunk::AbstractStreamChunk; kwargs...)

Extract the content from the chunk for Google Gemini API.
"""
function extract_content(::GoogleStream, chunk::AbstractStreamChunk; kwargs...)
    if !isnothing(chunk.json) && haskey(chunk.json, :candidates)
        candidates = chunk.json.candidates
        if length(candidates) > 0 && haskey(candidates[1], :content) && 
           haskey(candidates[1].content, :parts) && length(candidates[1].content.parts) > 0
            part = candidates[1].content.parts[1]
            if haskey(part, :text)
                return part.text
            end
        end
    end
    return nothing
end
"""
    build_response_body(::GoogleStream, cb::AbstractStreamCallback; kwargs...)

Build the response body from the chunks for Google Gemini API.
Returns an OpenAI-compatible response format to ensure compatibility with code expecting OpenAI responses.
"""
function build_response_body(::GoogleStream, cb::AbstractStreamCallback; kwargs...)
    # Extract all non-empty chunks with JSON data
    valid_chunks = filter(c -> !isnothing(c.json), cb.chunks)
    
    if isempty(valid_chunks)
        return nothing
    end
    
    # Use the last chunk as the base for our response
    last_chunk = valid_chunks[end].json
    
    # Combine text from all chunks
    combined_text = ""
    for chunk in valid_chunks
        if haskey(chunk.json, :candidates) && length(chunk.json.candidates) > 0 &&
           haskey(chunk.json.candidates[1], :content) && 
           haskey(chunk.json.candidates[1].content, :parts) && 
           length(chunk.json.candidates[1].content.parts) > 0 &&
           haskey(chunk.json.candidates[1].content.parts[1], :text)
            combined_text *= chunk.json.candidates[1].content.parts[1].text
        end
    end
    
    # Create an OpenAI-compatible response
    openai_resp = Dict{Symbol, Any}(
        :choices => [],
        :created => round(Int, time()),
        :model => get(last_chunk, :modelVersion, "gemini"),
        :object => "chat.completion",
        :usage => Dict{Symbol, Any}()
    )
    
    # Extract usage information
    if haskey(last_chunk, :usageMetadata)
        usage = last_chunk.usageMetadata
        openai_resp[:usage] = Dict{Symbol, Any}(
            :prompt_tokens => get(usage, :promptTokenCount, 0),
            :completion_tokens => get(usage, :candidatesTokenCount, 0),
            :total_tokens => get(usage, :totalTokenCount, 0)
        )
    end
    
    # Add the choice with the combined text
    if haskey(last_chunk, :candidates) && !isempty(last_chunk.candidates)
        finish_reason = get(last_chunk.candidates[1], :finishReason, "stop")
        choice = Dict{Symbol, Any}(
            :index => 0,
            :finish_reason => lowercase(finish_reason),
            :message => Dict{Symbol, Any}(
                :role => "assistant",
                :content => combined_text
            )
        )
        push!(openai_resp[:choices], choice)
    end
    
    return openai_resp
end