# Calling OpenAI with StreamCallbacks
using HTTP, JSON3
using StreamCallbacks

## Prepare target and auth
url = "https://api.openai.com/v1/chat/completions"
headers = [
    "Content-Type" => "application/json",
    "Authorization" => "Bearer $(get(ENV, "OPENAI_API_KEY", ""))"
];

## Send the request
cb = StreamCallback(; out = stdout, flavor = OpenAIStream())
messages = [Dict("role" => "user",
    "content" => "Count from 1 to 100.")]
payload = IOBuffer()
JSON3.write(payload,
    (; stream = true, messages, model = "gpt-4o-mini",
        stream_options = (; include_usage = true)))
resp = streamed_request!(cb, url, headers, payload);

## Check the response
resp # should be a `HTTP.Response` object with a message body like if we wouldn't use streaming

## Check the callback
cb.chunks # should be a vector of `StreamChunk` objects, each with a `json` field with received data from the API

# TIP: For debugging, use `cb.verbose = true` in the `StreamCallback` constructor to get more details on each chunk and enable DEBUG loglevel.
