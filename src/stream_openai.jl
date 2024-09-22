# Custom methods for OpenAI streaming -- flavor=OpenAIStream()
"""
    is_done(flavor::OpenAIStream, chunk::AbstractStreamChunk; kwargs...)

Check if the streaming is done. Shared by all streaming flavors currently.
"""
@inline function is_done(flavor::OpenAIStream, chunk::AbstractStreamChunk; kwargs...)
    chunk.data == "[DONE]"
end

"""
    extract_content(flavor::OpenAIStream, chunk::AbstractStreamChunk; kwargs...)

Extract the content from the chunk.
"""
@inline function extract_content(
        flavor::OpenAIStream, chunk::AbstractStreamChunk; kwargs...)
    if !isnothing(chunk.json)
        ## Can contain more than one choice for multi-sampling, but ignore for callback
        ## Get only the first choice
        choices = get(chunk.json, :choices, [])
        first_choice = get(choices, 1, Dict())
        delta = get(first_choice, :delta, Dict())
        out = get(delta, :content, nothing)
    else
        nothing
    end
end

"""
    build_response_body(flavor::OpenAIStream, cb::AbstractStreamCallback; verbose::Bool = false, kwargs...)

Build the response body from the chunks to mimic receiving a standard response from the API.

Note: Limited functionality for now. Does NOT support tool use, refusals, logprobs. Use standard responses for these.
"""
function build_response_body(
        flavor::OpenAIStream, cb::AbstractStreamCallback; verbose::Bool = false, kwargs...)
    isempty(cb.chunks) && return nothing
    response = nothing
    usage = nothing
    choices_output = Dict{Int, Dict{Symbol, Any}}()
    for i in eachindex(cb.chunks)
        chunk = cb.chunks[i]
        ## validate that we can access choices
        isnothing(chunk.json) && continue
        !haskey(chunk.json, :choices) && continue
        if isnothing(response)
            ## do it only once the first time when we have the json
            response = chunk.json |> copy
        end
        if isnothing(usage)
            usage_values = get(chunk.json, :usage, nothing)
            if !isnothing(usage_values)
                usage = usage_values |> copy
            end
        end
        for choice in chunk.json.choices
            index = get(choice, :index, nothing)
            isnothing(index) && continue
            if !haskey(choices_output, index)
                choices_output[index] = Dict{Symbol, Any}(:index => index)
            end
            index_dict = choices_output[index]
            finish_reason = get(choice, :finish_reason, nothing)
            if !isnothing(finish_reason)
                index_dict[:finish_reason] = finish_reason
            end
            ## skip for now
            # logprobs = get(choice, :logprobs, nothing)
            # if !isnothing(logprobs)
            #     choices_dict[index][:logprobs] = logprobs
            # end
            choice_delta = get(choice, :delta, Dict{Symbol, Any}())
            message_dict = get(index_dict, :message, Dict{Symbol, Any}(:content => ""))
            role = get(choice_delta, :role, nothing)
            if !isnothing(role)
                message_dict[:role] = role
            end
            content = get(choice_delta, :content, nothing)
            if !isnothing(content)
                message_dict[:content] *= content
            end
            ## skip for now
            # refusal = get(choice_delta, :refusal, nothing)
            # if !isnothing(refusal)
            #     message_dict[:refusal] = refusal
            # end
            index_dict[:message] = message_dict
        end
    end
    ## We know we have at least one chunk, let's use it for final response
    if !isnothing(response)
        # flatten the choices_dict into an array
        choices = [choices_output[index] for index in sort(collect(keys(choices_output)))]
        # overwrite the old choices
        response[:choices] = choices
        response[:object] = "chat.completion"
        response[:usage] = usage
    end
    return response
end
