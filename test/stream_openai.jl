
@testset "extract_content" begin
    # Test OpenAIStream
    openai_flavor = PT.OpenAIStream()

    # Test with valid JSON content
    valid_json_chunk = PT.StreamChunk(
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
    @test PT.extract_content(openai_flavor, valid_json_chunk) == "Hello, world!"

    # Test with empty choices
    empty_choices_chunk = PT.StreamChunk(
        json = JSON3.read("""
        {
            "choices": []
        }
        """)
    )
    @test isnothing(PT.extract_content(openai_flavor, empty_choices_chunk))

    # Test with missing delta
    missing_delta_chunk = PT.StreamChunk(
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
    @test isnothing(PT.extract_content(openai_flavor, missing_delta_chunk))

    # Test with missing content in delta
    missing_content_chunk = PT.StreamChunk(
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
    @test isnothing(PT.extract_content(openai_flavor, missing_content_chunk))

    # Test with non-JSON chunk
    non_json_chunk = PT.StreamChunk(data = "Plain text")
    @test isnothing(PT.extract_content(openai_flavor, non_json_chunk))

    # Test AnthropicStream
    anthropic_flavor = PT.AnthropicStream()

    # Test with valid content block
    valid_anthropic_chunk = PT.StreamChunk(
        json = JSON3.read("""
        {
            "content_block": {
                "text": "Hello from Anthropic!"
            }
        }
        """)
    )
    @test PT.extract_content(anthropic_flavor, valid_anthropic_chunk) ==
          "Hello from Anthropic!"

    # Test with valid delta
    valid_delta_chunk = PT.StreamChunk(
        json = JSON3.read("""
        {
            "delta": {
                "text": "Delta text"
            }
        }
        """)
    )
    @test PT.extract_content(anthropic_flavor, valid_delta_chunk) == "Delta text"

    # Test with missing text in content block
    missing_text_chunk = PT.StreamChunk(
        json = JSON3.read("""
        {
            "content_block": {
                "type": "text"
            }
        }
        """)
    )
    @test isnothing(PT.extract_content(anthropic_flavor, missing_text_chunk))

    # Test with non-zero index (should return nothing)
    non_zero_index_chunk = PT.StreamChunk(
        json = JSON3.read("""
        {
            "index": 1,
            "content_block": {
                "text": "This should be ignored"
            }
        }
        """)
    )
    @test isnothing(PT.extract_content(anthropic_flavor, non_zero_index_chunk))

    # Test with non-JSON chunk for Anthropic
    non_json_anthropic_chunk = PT.StreamChunk(data = "Plain Anthropic text")
    @test isnothing(PT.extract_content(anthropic_flavor, non_json_anthropic_chunk))

    # Test with unsupported flavor
    struct UnsupportedFlavor <: PT.AbstractStreamFlavor end
    unsupported_flavor = UnsupportedFlavor()
    @test_throws ArgumentError PT.extract_content(unsupported_flavor, PT.StreamChunk())
end

@testset "extract_content" begin
    ### OpenAIStream
    # Test case 1: Valid JSON with content
    valid_chunk = PT.StreamChunk(
        nothing,
        """{"choices":[{"delta":{"content":"Hello"}}]}""",
        JSON3.read("""{"choices":[{"delta":{"content":"Hello"}}]}""")
    )
    @test PT.extract_content(PT.OpenAIStream(), valid_chunk) == "Hello"

    # Test case 2: Valid JSON without content
    no_content_chunk = PT.StreamChunk(
        nothing,
        """{"choices":[{"delta":{}}]}""",
        JSON3.read("""{"choices":[{"delta":{}}]}""")
    )
    @test isnothing(PT.extract_content(PT.OpenAIStream(), no_content_chunk))

    # Test case 3: Valid JSON with empty content
    empty_content_chunk = PT.StreamChunk(
        nothing,
        """{"choices":[{"delta":{"content":""}}]}""",
        JSON3.read("""{"choices":[{"delta":{"content":""}}]}""")
    )
    @test PT.extract_content(PT.OpenAIStream(), empty_content_chunk) == ""

    # Test case 4: Invalid JSON structure
    invalid_chunk = PT.StreamChunk(
        nothing,
        """{"invalid":"structure"}""",
        JSON3.read("""{"invalid":"structure"}""")
    )
    @test isnothing(PT.extract_content(PT.OpenAIStream(), invalid_chunk))

    # Test case 5: Chunk with non-JSON data
    non_json_chunk = PT.StreamChunk(
        nothing,
        "This is not JSON",
        nothing
    )
    @test isnothing(PT.extract_content(PT.OpenAIStream(), non_json_chunk))

    # Test case 6: Multiple choices (should still return first choice)
    multiple_choices_chunk = PT.StreamChunk(
        nothing,
        """{"choices":[{"delta":{"content":"First"}},{"delta":{"content":"Second"}}]}""",
        JSON3.read("""{"choices":[{"delta":{"content":"First"}},{"delta":{"content":"Second"}}]}""")
    )
    @test PT.extract_content(PT.OpenAIStream(), multiple_choices_chunk) == "First"

    ### AnthropicStream
    # Test case 1: Valid JSON with content in content_block
    valid_chunk = PT.StreamChunk(
        nothing,
        """{"index":0,"content_block":{"text":"Hello from Anthropic"}}""",
        JSON3.read("""{"index":0,"content_block":{"text":"Hello from Anthropic"}}""")
    )
    @test PT.extract_content(PT.AnthropicStream(), valid_chunk) == "Hello from Anthropic"

    # Test case 2: Valid JSON with content in delta
    delta_chunk = PT.StreamChunk(
        nothing,
        """{"index":0,"delta":{"text":"Delta content"}}""",
        JSON3.read("""{"index":0,"delta":{"text":"Delta content"}}""")
    )
    @test PT.extract_content(PT.AnthropicStream(), delta_chunk) == "Delta content"

    # Test case 3: Valid JSON without text in content_block
    no_text_chunk = PT.StreamChunk(
        nothing,
        """{"index":0,"content_block":{"type":"text"}}""",
        JSON3.read("""{"index":0,"content_block":{"type":"text"}}""")
    )
    @test isnothing(PT.extract_content(PT.AnthropicStream(), no_text_chunk))

    # Test case 4: Valid JSON with non-zero index
    non_zero_index_chunk = PT.StreamChunk(
        nothing,
        """{"index":1,"content_block":{"text":"Should be ignored"}}""",
        JSON3.read("""{"index":1,"content_block":{"text":"Should be ignored"}}""")
    )
    @test isnothing(PT.extract_content(PT.AnthropicStream(), non_zero_index_chunk))

    # Test case 5: Chunk with non-JSON data
    non_json_chunk = PT.StreamChunk(
        nothing,
        "This is not JSON",
        nothing
    )
    @test isnothing(PT.extract_content(PT.AnthropicStream(), non_json_chunk))

    # Test case 6: Valid JSON with empty content
    empty_content_chunk = PT.StreamChunk(
        nothing,
        """{"index":0,"content_block":{"text":""}}""",
        JSON3.read("""{"index":0,"content_block":{"text":""}}""")
    )
    @test PT.extract_content(PT.AnthropicStream(), empty_content_chunk) == ""

    # Test case 7: Unknown flavor
    struct UnknownFlavor <: PT.AbstractStreamFlavor end
    unknown_flavor = UnknownFlavor()
    unknown_chunk = PT.StreamChunk(
        nothing,
        """{"content": "Test content"}""",
        JSON3.read("""{"content": "Test content"}""")
    )
    @test_throws ArgumentError PT.extract_content(unknown_flavor, unknown_chunk)
end

@testset "build_response_body-OpenAIStream" begin
    # Test case 1: Empty chunks
    cb_empty = PT.StreamCallback()
    response = PT.build_response_body(PT.OpenAIStream(), cb_empty)
    @test isnothing(response)

    # Test case 2: Single complete chunk
    cb_single = PT.StreamCallback()
    push!(cb_single.chunks,
        PT.StreamChunk(
            nothing,
            """{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}""",
            JSON3.read("""{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}""")
        ))
    response = PT.build_response_body(PT.OpenAIStream(), cb_single)
    @test response[:id] == "chatcmpl-123"
    @test response[:object] == "chat.completion"
    @test response[:model] == "gpt-4"
    @test length(response[:choices]) == 1
    @test response[:choices][1][:index] == 0
    @test response[:choices][1][:message][:role] == "assistant"
    @test response[:choices][1][:message][:content] == "Hello"

    # Test case 3: Multiple chunks forming a complete response
    cb_multiple = PT.StreamCallback()
    push!(cb_multiple.chunks,
        PT.StreamChunk(
            nothing,
            """{"id":"chatcmpl-456","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}""",
            JSON3.read("""{"id":"chatcmpl-456","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}""")
        ))
    push!(cb_multiple.chunks,
        PT.StreamChunk(
            nothing,
            """{"id":"chatcmpl-456","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}""",
            JSON3.read("""{"id":"chatcmpl-456","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}""")
        ))
    push!(cb_multiple.chunks,
        PT.StreamChunk(
            nothing,
            """{"id":"chatcmpl-456","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}""",
            JSON3.read("""{"id":"chatcmpl-456","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}""")
        ))
    response = PT.build_response_body(PT.OpenAIStream(), cb_multiple)
    @test response[:id] == "chatcmpl-456"
    @test response[:object] == "chat.completion"
    @test response[:model] == "gpt-4"
    @test length(response[:choices]) == 1
    @test response[:choices][1][:index] == 0
    @test response[:choices][1][:message][:role] == "assistant"
    @test response[:choices][1][:message][:content] == "Hello world"
    @test response[:choices][1][:finish_reason] == "stop"

    # Test case 4: Multiple choices
    cb_multi_choice = PT.StreamCallback()
    push!(cb_multi_choice.chunks,
        PT.StreamChunk(
            nothing,
            """{"id":"chatcmpl-789","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"First"},"finish_reason":null},{"index":1,"delta":{"role":"assistant","content":"Second"},"finish_reason":null}]}""",
            JSON3.read("""{"id":"chatcmpl-789","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"First"},"finish_reason":null},{"index":1,"delta":{"role":"assistant","content":"Second"},"finish_reason":null}]}""")
        ))
    response = PT.build_response_body(PT.OpenAIStream(), cb_multi_choice)
    @test response[:id] == "chatcmpl-789"
    @test length(response[:choices]) == 2
    @test response[:choices][1][:index] == 0
    @test response[:choices][1][:message][:content] == "First"
    @test response[:choices][2][:index] == 1
    @test response[:choices][2][:message][:content] == "Second"

    # Test case 5: Usage information
    cb_usage = PT.StreamCallback()
    push!(cb_usage.chunks,
        PT.StreamChunk(
            nothing,
            """{"id":"chatcmpl-101112","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"Test"},"finish_reason":null}],"usage":{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11}}""",
            JSON3.read("""{"id":"chatcmpl-101112","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"Test"},"finish_reason":null}],"usage":{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11}}""")
        ))
    response = PT.build_response_body(PT.OpenAIStream(), cb_usage)
    @test response[:usage][:prompt_tokens] == 10
    @test response[:usage][:completion_tokens] == 1
    @test response[:usage][:total_tokens] == 11
end