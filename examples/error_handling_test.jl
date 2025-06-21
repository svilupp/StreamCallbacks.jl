# Test error handling with custom IO that fails on "5"
using HTTP, JSON3
using StreamCallbacks
using StreamCallbacks: OpenAIStream, libcurl_streamed_request!

# Prepare target and auth
url = "https://api.openai.com/v1/chat/completions"
headers = [
    "Content-Type" => "application/json",
    "Authorization" => "Bearer $(get(ENV, "OPENAI_API_KEY", ""))"
]

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

messages = [Dict("role" => "user", "content" => "Count from 1 to 10.")]
payload = IOBuffer()
JSON3.write(payload, (; stream = true, messages, model = "gpt-4o-mini", stream_options = (; include_usage = true)))
payload_str = String(take!(payload))

println("=== Testing Error Handling ===")

# Test 1: HTTP.jl with error handling
println("\n1. Testing HTTP.jl error handling...")
cb_http = StreamCallback(; out = ErrorOnFiveIO(), flavor = OpenAIStream(), throw_on_error = true)
    resp_http = streamed_request!(cb_http, url, headers, IOBuffer(payload_str))
    println("HTTP: No error occurred (unexpected)")

# Test 2: LibCURL with error handling
println("\n2. Testing LibCURL error handling...")
cb_curl = StreamCallback(; out = ErrorOnFiveIO(), flavor = OpenAIStream(), throw_on_error = true)

resp_curl = libcurl_streamed_request!(cb_curl, url, headers, payload_str)
println("LibCURL: No error occurred (unexpected)")

println("\n=== Error Handling Test Complete ===")