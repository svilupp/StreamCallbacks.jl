# Calling OpenAI with StreamCallbacks
using HTTP, JSON3
using StreamCallbacks
using StreamCallbacks: OpenAIStream
using StreamCallbacks: libcurl_streamed_request!

# Prepare target and auth
url = "https://api.openai.com/v1/chat/completions"
headers = [
    "Content-Type" => "application/json",
    "Authorization" => "Bearer $(get(ENV, "OPENAI_API_KEY", ""))"
];
# Custom IO type that throws when it sees "5"
struct ErrorOnFiveIO <: IO
    buffer::Vector{String}
end
ErrorOnFiveIO() = ErrorOnFiveIO(String[])

function StreamCallbacks.print_content(out::ErrorOnFiveIO, text::AbstractString; kwargs...)
    push!(out.buffer, text)
    if occursin("5", text)
        error("Custom IO error: Found forbidden number '5' in: $(text)")
    end
end

# Send the request
# cb = StreamCallback(; out = stdout, flavor = OpenAIStream(), throw_on_error = false)
cb = StreamCallback(; out = ErrorOnFiveIO(), flavor = OpenAIStream(), throw_on_error = true)
very_long_text = ["(Just some random text $i.) " for i in 1:100_000] |> join
# very_long_text = ""
messages = [Dict("role" => "user", "content" => very_long_text * "Count from 1 to 10.")]
using LLMRateLimiters
# @show LLMRateLimiters.estimate_tokens(messages[1]["content"])

#
payload = IOBuffer()
JSON3.write(payload,
    (; stream = true, messages, model = "gpt-4o-mini",
        stream_options = (; include_usage = true)))

# Test different streaming methods:
# 1. HTTP.jl based (default)
# payload_str = String(take!(payload))
# resp = @time streamed_request!(cb, url, headers, IOBuffer(payload_str));
# @show resp

# 2. Socket-based streaming
# resp = socket_streamed_request!(cb, url, headers, String(take!(payload)));

# 3. LibCURL-based streaming (recommended)
# Clear chunks from previous test to avoid accumulation
empty!(cb.chunks)
resp = @time libcurl_streamed_request!(cb, url, headers, payload);
@show resp
;
## Check the response
resp # should be a `HTTP.Response` object with a message body like if we wouldn't use streaming

## Check the callback
cb.chunks # should be a vector of `StreamChunk` objects, each with a `json` field with received data from the API

# TIP: For debugging, use `cb.verbose = true` in the `StreamCallback` constructor to get more details on each chunk and enable DEBUG loglevel.
