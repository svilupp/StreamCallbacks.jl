
# ## Default methods

"""
    extract_chunks(flavor::AbstractStreamFlavor, blob::AbstractString;
        spillover::AbstractString = "", verbose::Bool = false, kwargs...)

Extract the chunks from the received SSE blob. Correctly implements SSE spec field parsing.
"""
@inline function extract_chunks(flavor::AbstractStreamFlavor, blob::AbstractString;
        spillover::AbstractString = "", verbose::Bool = false, kwargs...)
    
    # Handle any spillover from previous incomplete message
    full_blob = spillover * blob
    
    # Split on double newlines (SSE message boundaries)
    messages = split(full_blob, r"\n\n")
    
    # Check if last message is incomplete (no trailing \n\n)
    next_spillover = ""
    if !endswith(full_blob, "\n\n") && !isempty(messages)
        # Last message might be incomplete, save it for next time
        next_spillover = pop!(messages)
        verbose && @info "Incomplete message detected, spillover: $(repr(next_spillover))"
    end
    
    chunks = StreamChunk[]
    
    for message in messages
        isempty(strip(message)) && continue
        
        # Parse line starts
        event_name = nothing
        data_parts = String[]
        
        for line in split(message, '\n')
            line = rstrip(line, '\r')  # Handle \r\n
            if startswith(line, "data:")
                # SSE spec: collect characters after first colon, remove leading space if present
                field_value = line[6:end]  # Everything after "data:"
                if startswith(field_value, " ")
                    field_value = field_value[2:end]  # Remove leading space per SSE spec
                end
                push!(data_parts, field_value)
            elseif startswith(line, "event:")
                # Same rule applies for event field
                field_value = line[7:end]  # Everything after "event:"
                if startswith(field_value, " ")
                    field_value = field_value[2:end]  # Remove leading space per SSE spec
                end
                event_name = Symbol(field_value)
            end
            # Ignore other SSE fields like id:, retry:, comments
        end
        
        isempty(data_parts) && continue
        
        # Join multiple data lines with newlines (SSE spec)
        raw_data = join(data_parts, '\n')
        # SSE spec: remove final trailing newline if present
        if endswith(raw_data, '\n')
            raw_data = raw_data[1:end-1]
        end
        
        # More robust JSON detection - handle both objects and arrays
        parsed_json = if !isempty(strip(raw_data))
            stripped = strip(raw_data)
            is_json = (startswith(stripped, '{') && endswith(stripped, '}')) ||
                     (startswith(stripped, '[') && endswith(stripped, ']'))
            if is_json
                try
                    JSON3.read(raw_data)
                catch e
                    verbose && @warn "Cannot parse JSON: $(repr(raw_data))"
                    nothing
                end
            else
                nothing
            end
        else
            nothing
        end
        
        push!(chunks, StreamChunk(event_name, raw_data, parsed_json))
    end
    
    return chunks, next_spillover
end

function is_done(flavor::AbstractStreamFlavor, chunk::AbstractStreamChunk; kwargs...)
    throw(ArgumentError("is_done is not implemented for flavor $(typeof(flavor))"))
end

function extract_content(
        flavor::AbstractStreamFlavor, chunk::AbstractStreamChunk; kwargs...)
    throw(ArgumentError("extract_content is not implemented for flavor $(typeof(flavor))"))
end

function print_content(out::Any, text::AbstractString; kwargs...)
    throw(ArgumentError("print_content is not implemented for sink $(typeof(out))"))
end

"""
    print_content(out::IO, text::AbstractString; kwargs...)

Print the content to the IO output stream `out`.
"""
@inline function print_content(out::IO, text::AbstractString; kwargs...)
    print(out, text)
    # flush(stdout)
end
"""
    print_content(out::Channel, text::AbstractString; kwargs...)

Print the content to the provided Channel `out`.
"""
@inline function print_content(out::Channel, text::AbstractString; kwargs...)
    put!(out, text)
end

"""
    print_content(out::Nothing, text::Any)

Do nothing if the output stream is `nothing`.
"""
@inline function print_content(out::Nothing, text::Any; kwargs...)
    return nothing
end
print_content(::Nothing, ::AbstractString; kwargs...) = nothing

"""
    callback(cb::AbstractStreamCallback, chunk::AbstractStreamChunk; kwargs...)

Process the chunk to be printed and print it. It's a wrapper for two operations:
- extract the content from the chunk using `extract_content`
- print the content to the output stream using `print_content`
"""
@inline function callback(cb::AbstractStreamCallback, chunk::AbstractStreamChunk; kwargs...)
    processed_text = extract_content(cb.flavor, chunk; kwargs...)
    isnothing(processed_text) && return nothing
    print_content(cb.out, processed_text; kwargs...)
    return nothing
end

"""
    handle_error_message(chunk::AbstractStreamChunk; throw_on_error::Bool = false, kwargs...)

Handles error messages from the streaming response.
"""
@inline function handle_error_message(
        chunk::AbstractStreamChunk; throw_on_error::Bool = false, kwargs...)
    if chunk.event == :error ||
       (isnothing(chunk.event) && !isnothing(chunk.json) &&
        haskey(chunk.json, :error))
        has_error_dict = !isnothing(chunk.json) &&
                         get(chunk.json, :error, nothing) isa AbstractDict
        ## Build the error message
        error_str = if has_error_dict
            join(
                ["$(titlecase(string(k))): $(v)"
                 for (k, v) in pairs(chunk.json.error)],
                ", ")
        else
            string(chunk.data)
        end
        ## Define whether to throw an error
        error_msg = "Error detected in the streaming response: $(error_str)"
        if throw_on_error
            throw(Exception(error_msg))
        else
            @warn error_msg
        end
    end
    return nothing
end

"""
    streamed_request!(cb::AbstractStreamCallback, url, headers, input; kwargs...)

End-to-end wrapper for POST streaming requests. 
In-place modification of the callback object (`cb.chunks`) with the results of the request being returned.
We build the `body` of the response object in the end and write it into the `resp.body`.

Returns the response object.

# Arguments
- `cb`: The callback object.
- `url`: The URL to send the request to.
- `headers`: The headers to send with the request.
- `input`: A buffer with the request body.
- `kwargs`: Additional keyword arguments.
"""
function streamed_request!(cb::AbstractStreamCallback, url, headers, input; kwargs...)
    verbose = get(kwargs, :verbose, false) || cb.verbose
    resp = HTTP.open("POST", url, headers; kwargs...) do stream
        write(stream, String(take!(input)))
        HTTP.closewrite(stream)
        response = HTTP.startread(stream)

        ## Content type must be text/event-stream
        content_type = [header[2]
                        for header in response.headers
                        if lowercase(header[1]) == "content-type"]
        @assert length(content_type)==1 "Content-Type header must be present and unique"
        if cb.flavor isa OllamaStream
            ## Provide barebone support for application/x-ndjson (Ollama) -- it will emit "buffer spillover warnings
            @assert occursin(
                "application/x-ndjson", lowercase(content_type[1])) "For OllamaStream flavor, Content-Type must be application/x-ndjson"
        else
            ## For non-ollama streams, we accept only text/event-stream
            @assert occursin(
                "text/event-stream", lowercase(content_type[1])) """
                Content-Type header should include the type text/event-stream.
                Received type: $(content_type[1])
                Status code: $(response.status)
                Response headers:\n - $(join(["$k: $v" for (k,v) in response.headers], "\n - "))
                Response body: $(String(response.body))
                Please check the model you are using and that you set `stream=true`.
                """
        end

        isdone = false
        ## messages might be incomplete, so we need to keep track of the spillover
        spillover = ""
        while !eof(stream) || !isdone
            masterchunk = String(readavailable(stream))
            chunks, spillover = extract_chunks(
                cb.flavor, masterchunk; verbose, spillover, cb.kwargs...)

            for chunk in chunks
                ## Note you must have debug logging enabled to see this
                verbose && @debug "Chunk Data: $(chunk.data)"
                ## look for errors
                handle_error_message(chunk; cb.throw_on_error, verbose, cb.kwargs...)
                ## look for termination signal, but process all remaining chunks first
                is_done(cb.flavor, chunk; verbose, cb.kwargs...) && (isdone = true)
                ## trigger callback
                callback(cb, chunk; verbose, cb.kwargs...)
                ## Write into our CB chunks (for later processing)
                push!(cb, chunk)
            end
        end
        HTTP.closeread(stream)
    end
    ## For estetic reasons, if printing to stdout, we send a newline and flush
    cb.out == stdout && (println(); flush(stdout))

    body = build_response_body(cb.flavor, cb; verbose, cb.kwargs...)
    resp.body = JSON3.write(body)

    return resp
end
