# Calling Anthropic with StreamCallbacks
using HTTP, JSON3
using StreamCallbacks

## Prepare target and auth
url = "https://api.anthropic.com/v1/messages"
headers = [
    "anthropic-version" => "2023-06-01",
    "x-api-key" => "$(get(ENV, "ANTHROPIC_API_KEY", ""))"
];

## Send the request
cb = StreamCallback(; out = stdout, flavor = AnthropicStream())
messages = [Dict("role" => "user",
    "content" => "Count from 1 to 10. Start with numbers only.")]
payload = IOBuffer()
JSON3.write(payload,
    (; stream = true, messages, model = "claude-3-5-haiku-latest", max_tokens = 2048))
## , stop_sequences = ["2"]
resp = streamed_request!(cb, url, headers, payload);

## Check the response
resp # should be a `HTTP.Response` object with a message body like if we wouldn't use streaming

## Check the callback
cb.chunks # should be a vector of `StreamChunk` objects, each with a `json` field with received data from the API

# TIP: For debugging, use `cb.verbose = true` in the `StreamCallback` constructor to get more details on each chunk and enable DEBUG loglevel.

# This is the response body we get from the API but built together as if we didn't use streaming
StreamCallbacks.build_response_body(AnthropicStream(), cb)

# Show the thinking stream
cb = StreamCallback(;
    out = stdout, flavor = AnthropicStream(), kwargs = (; include_thinking = false))
messages = [Dict("role" => "user",
    "content" => "Count from 1 to 10000, skip prime numbers. Start with numbers only.")]
payload = IOBuffer()
JSON3.write(payload,
    (; stream = true, messages, model = "claude-3-7-haiku-20250219",
        max_tokens = 2500, thinking = Dict(:type => "enabled", :budget_tokens => 2000)))
resp = streamed_request!(cb, url, headers, payload);
