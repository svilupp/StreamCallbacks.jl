using Test
using HTTP
using JSON3
using StreamCallbacks

@testset "Anthropic Integration Test" begin
    # Skip if no API key is available
    api_key = get(ENV, "ANTHROPIC_API_KEY", "")
    if isempty(api_key)
        @test_skip "Skipping Anthropic integration test - no API key found"
        return
    end

    # Prepare target and auth
    url = "https://api.anthropic.com/v1/messages"
    headers = [
        "anthropic-version" => "2023-06-01",
        "x-api-key" => api_key
    ]

    # Test streaming callback
    cb = StreamCallback(; out = nothing, flavor = AnthropicStream())

    messages = [Dict("role" => "user", "content" => "Write me data: [1], data: [2], data: [3] ... do it 10 times.")]
    payload = IOBuffer()
    JSON3.write(payload,
        (; stream = true, messages, model = "claude-3-5-haiku-latest", max_tokens = 2048))

    # Send the request
    resp = streamed_request!(cb, url, headers, payload)

    # Test response structure - it's a NamedTuple with HTTP.Response fields
    @test resp.status == 200
    @test !isempty(cb.chunks)

    # Build response body
    response_body = StreamCallbacks.build_response_body(AnthropicStream(), cb)
    @test !isnothing(response_body)
    @test haskey(response_body, :content)
    @test length(response_body[:content]) >= 1
    @test response_body[:content][1][:type] == "text"

    # Extract the generated text
    generated_text = response_body[:content][1][:text]
    @test !isempty(generated_text)

    # Test that the response contains exactly 10 "data: " patterns
    # Count occurrences of "data: [" followed by a digit and "]"
    data_pattern_count = length(collect(eachmatch(r"data: \[\d+\]", generated_text)))
    @test data_pattern_count == 10  # Should be exactly 10

    # Alternative: count simple "data: " occurrences
    data_prefix_count = length(collect(eachmatch(r"data: ", generated_text)))
    @test data_prefix_count >= 10  # Should be at least 10

    # Test that chunks contain valid JSON
    json_chunks = filter(chunk -> !isnothing(chunk.json), cb.chunks)
    @test !isempty(json_chunks)

    # Test that we have message_start and message_stop events
    events = [chunk.event for chunk in cb.chunks if !isnothing(chunk.event)]
    @test :message_start in events
    @test :message_stop in events

    # Test content extraction from chunks
    content_chunks = String[]
    for chunk in cb.chunks
        content = extract_content(AnthropicStream(), chunk)
        if !isnothing(content) && !isempty(content)
            push!(content_chunks, content)
        end
    end
    @test !isempty(content_chunks)

    # Verify that concatenating content chunks gives us the full text
    reconstructed_text = join(content_chunks, "")
    @test reconstructed_text == generated_text

    # Debug output to see what we actually got
    println("Generated text: ", repr(generated_text))
    println("Data pattern count: ", data_pattern_count)
    println("Data prefix count: ", data_prefix_count)
end