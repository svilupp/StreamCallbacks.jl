
@testset "extract_content" begin
    # Test OpenAIStream
    openai_flavor = OpenAIStream()

    # Test with valid JSON content
    valid_json_chunk = StreamChunk(
        json = JSON3.read("""
        {
            "choices": [
                {
                    "delta": {
                        "content": "Hello, world!"
                    }
                }
            ]
        }
        """)
    )
    @test extract_content(openai_flavor, valid_json_chunk) == "Hello, world!"

    # Test with empty choices
    empty_choices_chunk = StreamChunk(
        json = JSON3.read("""
        {
            "choices": []
        }
        """)
    )
    @test isnothing(extract_content(openai_flavor, empty_choices_chunk))

    # Test with missing delta
    missing_delta_chunk = StreamChunk(
        json = JSON3.read("""
        {
            "choices": [
                {
                    "index": 0
                }
            ]
        }
        """)
    )
    @test isnothing(extract_content(openai_flavor, missing_delta_chunk))

    # Test with missing content in delta
    missing_content_chunk = StreamChunk(
        json = JSON3.read("""
        {
            "choices": [
                {
                    "delta": {
                        "role": "assistant"
                    }
                }
            ]
        }
        """)
    )
    @test isnothing(extract_content(openai_flavor, missing_content_chunk))

    # Test with non-JSON chunk
    non_json_chunk = StreamChunk(data = "Plain text")
    @test isnothing(extract_content(openai_flavor, non_json_chunk))

    # Test with unsupported flavor
    struct UnsupportedFlavor <: AbstractStreamFlavor end
    unsupported_flavor = UnsupportedFlavor()
    @test_throws ArgumentError extract_content(unsupported_flavor, StreamChunk())
end

@testset "extract_content" begin
    ### OpenAIStream
    # Test case 1: Valid JSON with content
    valid_chunk = StreamChunk(
        nothing,
        """{"choices":[{"delta":{"content":"Hello"}}]}""",
        JSON3.read("""{"choices":[{"delta":{"content":"Hello"}}]}""")
    )
    @test extract_content(OpenAIStream(), valid_chunk) == "Hello"

    # Test case 2: Valid JSON without content
    no_content_chunk = StreamChunk(
        nothing,
        """{"choices":[{"delta":{}}]}""",
        JSON3.read("""{"choices":[{"delta":{}}]}""")
    )
    @test isnothing(extract_content(OpenAIStream(), no_content_chunk))

    # Test case 3: Valid JSON with empty content
    empty_content_chunk = StreamChunk(
        nothing,
        """{"choices":[{"delta":{"content":""}}]}""",
        JSON3.read("""{"choices":[{"delta":{"content":""}}]}""")
    )
    @test extract_content(OpenAIStream(), empty_content_chunk) == ""

    # Test case 4: Invalid JSON structure
    invalid_chunk = StreamChunk(
        nothing,
        """{"invalid":"structure"}""",
        JSON3.read("""{"invalid":"structure"}""")
    )
    @test isnothing(extract_content(OpenAIStream(), invalid_chunk))

    # Test case 5: Chunk with non-JSON data
    non_json_chunk = StreamChunk(
        nothing,
        "This is not JSON",
        nothing
    )
    @test isnothing(extract_content(OpenAIStream(), non_json_chunk))

    # Test case 6: Multiple choices (should still return first choice)
    multiple_choices_chunk = StreamChunk(
        nothing,
        """{"choices":[{"delta":{"content":"First"}},{"delta":{"content":"Second"}}]}""",
        JSON3.read("""{"choices":[{"delta":{"content":"First"}},{"delta":{"content":"Second"}}]}""")
    )
    @test extract_content(OpenAIStream(), multiple_choices_chunk) == "First"

    # Test case 7: Unknown flavor
    struct UnknownFlavor <: AbstractStreamFlavor end
    unknown_flavor = UnknownFlavor()
    unknown_chunk = StreamChunk(
        nothing,
        """{"content": "Test content"}""",
        JSON3.read("""{"content": "Test content"}""")
    )
    @test_throws ArgumentError extract_content(unknown_flavor, unknown_chunk)
end

@testset "build_response_body-OpenAIStream" begin
    # Test case 1: Empty chunks
    cb_empty = StreamCallback()
    response = build_response_body(OpenAIStream(), cb_empty)
    @test isnothing(response)

    # Test case 2: Single complete chunk
    cb_single = StreamCallback()
    push!(cb_single.chunks,
        StreamChunk(
            nothing,
            """{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}""",
            JSON3.read("""{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}""")
        ))
    response = build_response_body(OpenAIStream(), cb_single)
    @test response[:id] == "chatcmpl-123"
    @test response[:object] == "chat.completion"
    @test response[:model] == "gpt-4"
    @test length(response[:choices]) == 1
    @test response[:choices][1][:index] == 0
    @test response[:choices][1][:message][:role] == "assistant"
    @test response[:choices][1][:message][:content] == "Hello"

    # Test case 3: Multiple chunks forming a complete response
    cb_multiple = StreamCallback()
    push!(cb_multiple.chunks,
        StreamChunk(
            nothing,
            """{"id":"chatcmpl-456","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}""",
            JSON3.read("""{"id":"chatcmpl-456","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}""")
        ))
    push!(cb_multiple.chunks,
        StreamChunk(
            nothing,
            """{"id":"chatcmpl-456","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}""",
            JSON3.read("""{"id":"chatcmpl-456","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}""")
        ))
    push!(cb_multiple.chunks,
        StreamChunk(
            nothing,
            """{"id":"chatcmpl-456","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}""",
            JSON3.read("""{"id":"chatcmpl-456","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}""")
        ))
    response = build_response_body(OpenAIStream(), cb_multiple)
    @test response[:id] == "chatcmpl-456"
    @test response[:object] == "chat.completion"
    @test response[:model] == "gpt-4"
    @test length(response[:choices]) == 1
    @test response[:choices][1][:index] == 0
    @test response[:choices][1][:message][:role] == "assistant"
    @test response[:choices][1][:message][:content] == "Hello world"
    @test response[:choices][1][:finish_reason] == "stop"

    # Test case 4: Multiple choices
    cb_multi_choice = StreamCallback()
    push!(cb_multi_choice.chunks,
        StreamChunk(
            nothing,
            """{"id":"chatcmpl-789","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"First"},"finish_reason":null},{"index":1,"delta":{"role":"assistant","content":"Second"},"finish_reason":null}]}""",
            JSON3.read("""{"id":"chatcmpl-789","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"First"},"finish_reason":null},{"index":1,"delta":{"role":"assistant","content":"Second"},"finish_reason":null}]}""")
        ))
    response = build_response_body(OpenAIStream(), cb_multi_choice)
    @test response[:id] == "chatcmpl-789"
    @test length(response[:choices]) == 2
    @test response[:choices][1][:index] == 0
    @test response[:choices][1][:message][:content] == "First"
    @test response[:choices][2][:index] == 1
    @test response[:choices][2][:message][:content] == "Second"

    # Test case 5: Usage information
    cb_usage = StreamCallback()
    push!(cb_usage.chunks,
        StreamChunk(
            nothing,
            """{"id":"chatcmpl-101112","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"Test"},"finish_reason":null}],"usage":{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11}}""",
            JSON3.read("""{"id":"chatcmpl-101112","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"Test"},"finish_reason":null}],"usage":{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11}}""")
        ))
    response = build_response_body(OpenAIStream(), cb_usage)
    @test response[:usage][:prompt_tokens] == 10
    @test response[:usage][:completion_tokens] == 1
    @test response[:usage][:total_tokens] == 11
end
