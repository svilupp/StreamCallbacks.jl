# Custom methods for Ollama streaming -- flavor=OllamaStream()
# works only for Ollama `api/chat` endpoint!!

@inline function extract_chunks(flavor::OllamaStream, blob::AbstractString;
        spillover::AbstractString = "", verbose::Bool = false, kwargs...)
    @assert spillover=="" "OllamaStream does not support spillover"
    chunks = StreamChunk[]
    ## Assumes application/x-ndjson format, not SSE!
    blob_split = split(blob, "\n\n")
    for (bi, chunk) in enumerate(blob_split)
        isempty(chunk) && continue
        event_name = nothing
        data = rstrip(chunk, '\n')
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
    return chunks, spillover
end

"Terminates the stream when the `done` key in the JSON object is `true`."
function is_done(flavor::OllamaStream, chunk::StreamChunk; kwargs...)
    !isnothing(chunk.json) && haskey(chunk.json, "done") && chunk.json["done"] == true
end
function extract_content(flavor::OllamaStream, chunk::StreamChunk; kwargs...)
    isnothing(chunk.json) && return nothing
    message = get(chunk.json, :message, Dict())
    out = get(message, :content, nothing)
end
function build_response_body(
        flavor::OllamaStream, cb::AbstractStreamCallback; verbose::Bool = false, kwargs...)
    isempty(cb.chunks) && return nothing
    response = nothing
    usage = nothing
    content = IOBuffer()
    for i in eachindex(cb.chunks)
        chunk = cb.chunks[i]
        ## validate that we can access choices
        isnothing(chunk.json) && continue
        !haskey(chunk.json, :message) && continue
        if isnothing(response)
            ## do it only once the first time when we have the json
            response = chunk.json |> copy
        end
        if isnothing(usage) && ((haskey(chunk.json, :prompt_eval_count) ||
             haskey(chunk.json, :eval_count)))
            fields = [
                :prompt_eval_count, :prompt_eval_duration, :eval_duration, :eval_count]
            usage = Dict(field => chunk.json[field]
            for field in fields if haskey(chunk.json, field))
        end
        message = chunk.json[:message]
        temp = get(message, :content, nothing)
        if !isnothing(temp)
            write(content, temp)
        end
    end
    ## We know we have at least one chunk, let's use it for final response
    if !isnothing(response) && haskey(response, :message)
        ## We need to make sure we're operating on a copy of the response
        response isa JSON3.Object && (response = copy(response))
        response[:message][:content] = String(take!(content))
        !isnothing(usage) && merge!(response, usage)
    end
    return response
end