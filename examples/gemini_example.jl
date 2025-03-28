# Calling Google Gemini with StreamCallbacks
using HTTP, JSON3
using StreamCallbacks

# Prepare target and auth
api_key = get(ENV, "GOOGLE_API_KEY", "")
model = "gemini-2.0-flash"
model = "gemini-2.5-pro-exp-03-25"
url = "https://generativelanguage.googleapis.com/v1beta/models/$model:streamGenerateContent?alt=sse&key=$api_key"
headers = ["Content-Type" => "application/json"]

# Prepare the request payload
cb = StreamCallback(; out = stdout, flavor = GoogleStream(), verbose = true)
payload = IOBuffer()
JSON3.write(payload, Dict(
    :contents => [Dict(
        :parts => [Dict(
            :text => "Count from 1 to 20."
        )]
    )]
))

# Send the request
resp = streamed_request!(cb, url, headers, payload)

## Check the response
resp # should be a `HTTP.Response` object with a message body like if we wouldn't use streaming

## Check the callback
cb.chunks # should be a vector of `StreamChunk` objects, each with a `json` field with received data from the API
