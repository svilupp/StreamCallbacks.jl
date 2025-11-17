# Calling Google AI with OpenAI Schema using StreamCallbacks
using HTTP, JSON3
using StreamCallbacks

# Prepare target and auth for Google AI Studio
url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
headers = [
    "Content-Type" => "application/json",
    "Authorization" => "Bearer $(get(ENV, "GOOGLE_API_KEY", ""))"
]

# Send the request with OpenAI-compatible format
cb = StreamCallback(; out = stdout, flavor = OpenAIStream())  # Use OpenAIStream for Google's OpenAI schema
messages = [Dict("role" => "user",
    "content" => "Count from 1 to 10. Start with numbers only.")]
payload = IOBuffer()
JSON3.write(payload,
    (; stream = true, messages, model = "gemini-2.5-flash", stream_options = (; include_usage = true)))

resp = streamed_request!(cb, url, headers, payload);

println("Response status: ", resp.status)
println("Collected chunks: ", length(cb.chunks))