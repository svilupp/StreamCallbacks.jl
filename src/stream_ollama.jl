# Custom methods for Ollama streaming -- flavor=OllamaStream()
# works only for Ollama `api/chat` endpoint!!

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
        response[:message][:content] = String(take!(content))
        !isnothing(usage) && merge!(response, usage)
    end
    return response
end