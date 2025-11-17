# Custom methods for OpenAI Response API streaming -- flavor=ResponseStream()

"""
    is_done(flavor::ResponseStream, chunk::AbstractStreamChunk; kwargs...)

Check if the streaming is done for Response API.
Response API sends "response.completed" event when done.
"""
@inline function is_done(flavor::ResponseStream, chunk::AbstractStreamChunk; kwargs...)
    !isnothing(chunk.json) && get(chunk.json, :type, "") == "response.completed"
end

"""
    extract_content(flavor::ResponseStream, chunk::AbstractStreamChunk; kwargs...)

Extract the content from Response API streaming chunks.
Response API uses event-based streaming with `response.output_text.delta` events.
"""
@inline function extract_content(
        flavor::ResponseStream, chunk::AbstractStreamChunk; kwargs...)
    if !isnothing(chunk.json)
        # Response API uses different structure: {"type":"response.output_text.delta", "delta":"text", ...}
        chunk_type = get(chunk.json, :type, "")
        
        # Handle regular output text deltas
        if chunk_type == "response.output_text.delta"
            return get(chunk.json, :delta, nothing)
        
        # Handle reasoning summary text deltas (for reasoning traces)
        elseif chunk_type == "response.reasoning_summary_text.delta"
            delta_text = get(chunk.json, :delta, nothing)
            if !isnothing(delta_text)
                # Italic reasoning summary segments
                return "\e[3m" * delta_text * "\e[23m"
            end
            return nothing

        # When reasoning summary text is done, emit a newline separator
        elseif chunk_type == "response.reasoning_summary_text.done"
            return "\n"
        end
    end
    return nothing
end

"""
    build_response_body(flavor::ResponseStream, cb::AbstractStreamCallback; verbose::Bool = false, kwargs...)

Build the response body from the chunks to mimic receiving a standard response from the API.
Reconstructs the Response API format from streaming chunks.
"""
function build_response_body(
        flavor::ResponseStream, cb::AbstractStreamCallback; verbose::Bool = false, kwargs...)
    isempty(cb.chunks) && return nothing
    
    response = nothing
    content_parts = String[]
    
    for chunk in cb.chunks
        isnothing(chunk.json) && continue
        
        chunk_type = get(chunk.json, :type, "")
        
        # Initialize response from the first response.created event
        if chunk_type == "response.created" && isnothing(response)
            response = get(chunk.json, :response, Dict()) |> copy
        end
        
        # Update response from response.completed event (final state)
        if chunk_type == "response.completed"
            final_response = get(chunk.json, :response, Dict())
            if !isnothing(response)
                # Merge the final response data
                response = merge(response, final_response)
            else
                response = final_response |> copy
            end
        end
        
        # Collect content from delta events
        if chunk_type == "response.output_text.delta"
            delta_content = get(chunk.json, :delta, "")
            if !isempty(delta_content)
                push!(content_parts, delta_content)
            end
        end
    end
    
    # If we have response but need to reconstruct content
    if !isnothing(response) && !isempty(content_parts)
        full_content = join(content_parts)
        
        # Ensure we have the output structure
        if !haskey(response, :output) || isempty(response[:output])
            # Create a basic message output structure
            response[:output] = [
                Dict(
                    :type => "message",
                    :status => "completed",
                    :content => [
                        Dict(
                            :type => "output_text",
                            :text => full_content
                        )
                    ],
                    :role => "assistant"
                )
            ]
        else
            # Convert output array to mutable and update existing output with reconstructed content
            output_array = []
            for output_item in response[:output]
                output_dict = Dict(output_item)  # Convert JSON3.Object to Dict
                if get(output_dict, :type, "") == "message"
                    content_array = []
                    for content_item in get(output_dict, :content, [])
                        content_dict = Dict(content_item)  # Convert JSON3.Object to Dict
                        if get(content_dict, :type, "") == "output_text"
                            content_dict[:text] = full_content
                        end
                        push!(content_array, content_dict)
                    end
                    output_dict[:content] = content_array
                end
                push!(output_array, output_dict)
            end
            response[:output] = output_array
        end
    end
    
    return response
end