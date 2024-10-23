@testset "is_done" begin
    # Test case 1: JSON with done = true
    done_chunk = StreamChunk(
        nothing,
        """{"done":true}""",
        JSON3.read("""{"done":true}""")
    )
    @test is_done(OllamaStream(), done_chunk) == true

    # Test case 2: JSON with done = false
    not_done_chunk = StreamChunk(
        nothing,
        """{"done":false}""",
        JSON3.read("""{"done":false}""")
    )
    @test is_done(OllamaStream(), not_done_chunk) == false

    # Test case 3: JSON without done key
    no_done_chunk = StreamChunk(
        nothing,
        """{"message":{"content":"Hello"}}""",
        JSON3.read("""{"message":{"content":"Hello"}}""")
    )
    @test is_done(OllamaStream(), no_done_chunk) == false

    # Test case 4: Non-JSON chunk
    non_json_chunk = StreamChunk(
        nothing,
        "This is not JSON",
        nothing
    )
    @test is_done(OllamaStream(), non_json_chunk) == false
end

@testset "extract_content" begin
    # Test case 1: Valid JSON with content
    valid_chunk = StreamChunk(
        nothing,
        """{"message":{"content":"Hello from Ollama"}}""",
        JSON3.read("""{"message":{"content":"Hello from Ollama"}}""")
    )
    @test extract_content(OllamaStream(), valid_chunk) == "Hello from Ollama"

    # Test case 2: JSON without message key
    no_message_chunk = StreamChunk(
        nothing,
        """{"other":"data"}""",
        JSON3.read("""{"other":"data"}""")
    )
    @test isnothing(extract_content(OllamaStream(), no_message_chunk))

    # Test case 3: JSON with empty content
    empty_content_chunk = StreamChunk(
        nothing,
        """{"message":{"content":""}}""",
        JSON3.read("""{"message":{"content":""}}""")
    )
    @test extract_content(OllamaStream(), empty_content_chunk) == ""

    # Test case 4: Non-JSON chunk
    non_json_chunk = StreamChunk(
        nothing,
        "This is not JSON",
        nothing
    )
    @test isnothing(extract_content(OllamaStream(), non_json_chunk))
end

@testset "build_response_body" begin
    # Test case 1: Empty chunks
    cb_empty = StreamCallback(flavor = OllamaStream())
    @test isnothing(build_response_body(OllamaStream(), cb_empty))

    # Test case 2: Single complete chunk
    cb_single = StreamCallback(flavor = OllamaStream())
    push!(cb_single.chunks,
        StreamChunk(
            nothing,
            """{"model":"ollama","created_at":"2023-11-06T12:34:56Z","message":{"role":"assistant","content":"Hello"},"done":true}""",
            JSON3.read("""{"model":"ollama","created_at":"2023-11-06T12:34:56Z","message":{"role":"assistant","content":"Hello"},"done":true}""")
        ))
    response = build_response_body(OllamaStream(), cb_single)
    @test response[:model] == "ollama"
    @test response[:message][:content] == "Hello"

    # Test case 3: Multiple chunks forming a complete response
    cb_multiple = StreamCallback(flavor = OllamaStream())
    push!(cb_multiple.chunks,
        StreamChunk(
            nothing,
            """{"model":"ollama","created_at":"2023-11-06T12:34:56Z","message":{"role":"assistant","content":"Hello"},"done":false}""",
            JSON3.read("""{"model":"ollama","created_at":"2023-11-06T12:34:56Z","message":{"role":"assistant","content":"Hello"},"done":false}""")
        ))
    push!(cb_multiple.chunks,
        StreamChunk(
            nothing,
            """{"model":"ollama","created_at":"2023-11-06T12:34:57Z","message":{"role":"assistant","content":" world"},"done":false}""",
            JSON3.read("""{"model":"ollama","created_at":"2023-11-06T12:34:57Z","message":{"role":"assistant","content":" world"},"done":false}""")
        ))
    push!(cb_multiple.chunks,
        StreamChunk(
            nothing,
            """{"model":"ollama","created_at":"2023-11-06T12:34:58Z","message":{"role":"assistant","content":"!"},"done":true}""",
            JSON3.read("""{"model":"ollama","created_at":"2023-11-06T12:34:58Z","message":{"role":"assistant","content":"!"},"done":true}""")
        ))
    response = build_response_body(OllamaStream(), cb_multiple)
    @test response[:model] == "ollama"
    @test response[:message][:content] == "Hello world!"

    # Test case 4: Chunks with usage information
    cb_usage = StreamCallback(flavor = OllamaStream())
    push!(cb_usage.chunks,
        StreamChunk(
            nothing,
            """{"model":"ollama","created_at":"2023-11-06T12:34:56Z","message":{"role":"assistant","content":"Test"},"done":true,"prompt_eval_count":10,"eval_count":5}""",
            JSON3.read("""{"model":"ollama","created_at":"2023-11-06T12:34:56Z","message":{"role":"assistant","content":"Test"},"done":true,"prompt_eval_count":10,"eval_count":5}""")
        ))
    response = build_response_body(OllamaStream(), cb_usage)
    @test response[:prompt_eval_count] == 10
    @test response[:eval_count] == 5
end
