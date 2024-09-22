
# ## Default methods

"""
    extract_chunks(flavor::AbstractStreamFlavor, blob::AbstractString;
        spillover::AbstractString = "", verbose::Bool = false, kwargs...)

Extract the chunks from the received SSE blob. Shared by all streaming flavors currently.

Returns a list of `StreamChunk` and the next spillover (if message was incomplete).
"""
@inline function extract_chunks(flavor::AbstractStreamFlavor, blob::AbstractString;
        spillover::AbstractString = "", verbose::Bool = false, kwargs...)
    chunks = StreamChunk[]
    next_spillover = ""
    ## SSE come separated by double-newlines
    blob_split = split(blob, "\n\n")
    for (bi, chunk) in enumerate(blob_split)
        isempty(chunk) && continue
        event_split = split(chunk, "event: ")
        has_event = length(event_split) > 1
        # if length>1, we know it was there!
        for event_blob in event_split
            isempty(event_blob) && continue
            event_name = nothing
            data_buf = IOBuffer()
            data_splits = split(event_blob, "data: ")
            for i in eachindex(data_splits)
                isempty(data_splits[i]) && continue
                if i == 1 & has_event && !isempty(data_splits[i])
                    ## we have an event name
                    event_name = strip(data_splits[i]) |> Symbol
                elseif bi == 1 && i == 1 && !isempty(data_splits[i])
                    ## in the first part of the first blob, it must be a spillover
                    spillover = string(spillover, rstrip(data_splits[i], '\n'))
                    verbose && @info "Buffer spillover detected: $(spillover)"
                elseif i > 1
                    ## any subsequent data blobs are accummulated into the data buffer
                    ## there can be multiline data that must be concatenated
                    data_chunk = rstrip(data_splits[i], '\n')
                    write(data_buf, data_chunk)
                end
            end

            ## Parse the spillover
            if bi == 1 && !isempty(spillover)
                data = spillover
                json = if startswith(data, '{') && endswith(data, '}')
                    try
                        JSON3.read(data)
                    catch e
                        verbose && @warn "Cannot parse JSON: $data"
                        nothing
                    end
                else
                    nothing
                end
                ## ignore event name
                push!(chunks, StreamChunk(; data = spillover, json = json))
                # reset the spillover
                spillover = ""
            end
            ## On the last iteration of the blob, check if we spilled over
            if bi == length(blob_split) && length(data_splits) > 1 &&
               !isempty(strip(data_splits[end]))
                verbose && @info "Incomplete message detected: $(data_splits[end])"
                next_spillover = String(take!(data_buf))
                ## Do not save this chunk
            else
                ## Try to parse the data as JSON
                data = String(take!(data_buf))
                isempty(data) && continue
                ## try to build a JSON object if it's a well-formed JSON string
                json = if startswith(data, '{') && endswith(data, '}')
                    try
                        JSON3.read(data)
                    catch e
                        verbose && @warn "Cannot parse JSON: $data"
                        nothing
                    end
                else
                    nothing
                end
                ## Create a new chunk
                push!(chunks, StreamChunk(event_name, data, json))
            end
        end
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
        @assert occursin("text/event-stream", content_type[1]) "Content-Type header include the type text/event-stream"

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