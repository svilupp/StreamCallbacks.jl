# Test long context with HTTP vs LibCURL performance comparison
using HTTP, JSON3
using StreamCallbacks
using StreamCallbacks: OpenAIStream, libcurl_streamed_request!

# Prepare target and auth
url = "https://api.openai.com/v1/chat/completions"
headers = [
    "Content-Type" => "application/json",
    "Authorization" => "Bearer $(get(ENV, "OPENAI_API_KEY", ""))"
]

# Create very long context
very_long_text = ["(Random text chunk $i.) " for i in 1:100_000] |> join
messages = [Dict("role" => "user", "content" => very_long_text * "Count from 1 to 3.")]

payload = IOBuffer()
JSON3.write(payload, (; stream = true, messages, model = "gpt-4o-mini", stream_options = (; include_usage = true)))
payload_str = String(take!(payload))

println("=== Testing Long Context Performance ===")
println("Context length: $(length(very_long_text)) characters")

# Test 1: HTTP.jl based streaming
println("\n1. Testing HTTP.jl streaming...")
cb_http = StreamCallback(; out = stdout, flavor = OpenAIStream(), throw_on_error = true)
@time begin
    resp_http = streamed_request!(cb_http, url, headers, IOBuffer(payload_str))
end
@show resp_http
println("HTTP chunks received: $(length(cb_http.chunks))")

# Test 2: LibCURL based streaming  
println("\n2. Testing LibCURL streaming...")
cb_curl = StreamCallback(; out = stdout, flavor = OpenAIStream(), throw_on_error = true)
@time begin
    resp_curl = libcurl_streamed_request!(cb_curl, url, headers, payload_str)
end
println("LibCURL chunks received: $(length(cb_curl.chunks))")

println("\n=== Performance Comparison Complete ===")