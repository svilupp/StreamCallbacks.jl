# Calling OpenAI Responses API with StreamCallbacks
using HTTP, JSON3
using StreamCallbacks

## Prepare target and auth
url = "https://api.openai.com/v1/responses"
headers = [
    "Content-Type" => "application/json",
    "Authorization" => "Bearer $(ENV["OPENAI_API_KEY"])"
];

## Send the request
cb = StreamCallback(; out = stdout, flavor = OpenAIResponsesStream())
payload = IOBuffer()
JSON3.write(payload, (; stream = true, input = "Count from 1 to 20", model = "gpt-5-mini"))
resp = streamed_request!(cb, url, headers, payload);

## Check the response
resp # should be a `HTTP.Response` object with a reconstructed response body

## Check the callback
cb.chunks # should be a vector of `StreamChunk` objects with received SSE events

# TIP: For debugging, use `cb.verbose = true` in the `StreamCallback` constructor and enable DEBUG loglevel.
