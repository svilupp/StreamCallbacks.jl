@testset "build_response_body-AnthropicStream" begin
    # Test case 1: Empty chunks
    cb_empty = StreamCallback(flavor = AnthropicStream())
    response = build_response_body(AnthropicStream(), cb_empty)
    @test isnothing(response)

    # Test case 2: Single message
    cb_single = StreamCallback(flavor = AnthropicStream())
    push!(cb_single.chunks,
        StreamChunk(
            :message_start,
            """{"message":{"content":[],"model":"claude-2","stop_reason":null,"stop_sequence":null}}""",
            JSON3.read("""{"message":{"content":[],"model":"claude-2","stop_reason":null,"stop_sequence":null}}""")
        ))
    response = build_response_body(AnthropicStream(), cb_single)
    @test response[:content][1][:type] == "text"
    @test response[:content][1][:text] == ""
    @test response[:model] == "claude-2"
    @test isnothing(response[:stop_reason])
    @test isnothing(response[:stop_sequence])

    # Test case 3: Multiple content blocks
    cb_multiple = StreamCallback(flavor = AnthropicStream())
    push!(cb_multiple.chunks,
        StreamChunk(
            :message_start,
            """{"message":{"content":[],"model":"claude-2","stop_reason":null,"stop_sequence":null}}""",
            JSON3.read("""{"message":{"content":[],"model":"claude-2","stop_reason":null,"stop_sequence":null}}""")
        ))
    push!(cb_multiple.chunks,
        StreamChunk(
            :content_block_start,
            """{"content_block":{"type":"text","text":"Hello"}}""",
            JSON3.read("""{"content_block":{"type":"text","text":"Hello"}}""")
        ))
    push!(cb_multiple.chunks,
        StreamChunk(
            :content_block_delta,
            """{"delta":{"type":"text","text":" world"}}""",
            JSON3.read("""{"delta":{"type":"text","text":" world"}}""")
        ))
    push!(cb_multiple.chunks,
        StreamChunk(
            :content_block_stop,
            """{"content_block":{"type":"text","text":"!"}}""",
            JSON3.read("""{"content_block":{"type":"text","text":"!"}}""")
        ))
    response = build_response_body(AnthropicStream(), cb_multiple)
    @test response[:content][1][:type] == "text"
    @test response[:content][1][:text] == "Hello world!"
    @test response[:model] == "claude-2"

    # Test case 4: With usage information
    cb_usage = StreamCallback(flavor = AnthropicStream())
    push!(cb_usage.chunks,
        StreamChunk(
            :message_start,
            """{"message":{"content":[],"model":"claude-2","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":5}}}""",
            JSON3.read("""{"message":{"content":[],"model":"claude-2","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":5}}}""")
        ))
    push!(cb_usage.chunks,
        StreamChunk(
            :content_block_start,
            """{"content_block":{"type":"text","text":"Test"}}""",
            JSON3.read("""{"content_block":{"type":"text","text":"Test"}}""")
        ))
    push!(cb_usage.chunks,
        StreamChunk(
            :message_delta,
            """{"delta":{"stop_reason": "end_turn"},"usage":{"output_tokens":7}}""",
            JSON3.read("""{"delta":{"stop_reason": "end_turn"},"usage":{"output_tokens":7}}""")
        ))
    response = build_response_body(AnthropicStream(), cb_usage)
    @test response[:content][1][:type] == "text"
    @test response[:content][1][:text] == "Test"
    @test response[:usage][:input_tokens] == 10
    @test response[:usage][:output_tokens] == 7
    @test response[:stop_reason] == "end_turn"

    # Test case 5: With stop reason
    cb_stop = StreamCallback(flavor = AnthropicStream())
    push!(cb_stop.chunks,
        StreamChunk(
            :message_start,
            """{"message":{"content":[],"model":"claude-2","stop_reason":null,"stop_sequence":null}}""",
            JSON3.read("""{"message":{"content":[],"model":"claude-2","stop_reason":null,"stop_sequence":null}}""")
        ))
    push!(cb_stop.chunks,
        StreamChunk(
            :content_block_start,
            """{"content_block":{"type":"text","text":"Final"}}""",
            JSON3.read("""{"content_block":{"type":"text","text":"Final"}}""")
        ))
    push!(cb_stop.chunks,
        StreamChunk(
            :message_delta,
            """{"delta":{"stop_reason":"max_tokens","stop_sequence":null}}""",
            JSON3.read("""{"delta":{"stop_reason":"max_tokens","stop_sequence":null}}""")
        ))
    response = build_response_body(AnthropicStream(), cb_stop)
    @test response[:content][1][:type] == "text"
    @test response[:content][1][:text] == "Final"
    @test response[:stop_reason] == "max_tokens"
    @test isnothing(response[:stop_sequence])
end

@testset "extract_content-AnthropicStream" begin
    # Test case 1: Nil JSON should return nothing
    chunk_nil = StreamChunk(:content_block_delta, "{}", nothing)
    @test isnothing(extract_content(AnthropicStream(), chunk_nil))

    # Test case 2: Non-content_block_delta chunk type should return nothing
    chunk_wrong_type = StreamChunk(
        :content_block_start,
        """{"type":"content_block_start","content_block":{"type":"text","text":"Hello"}}""",
        JSON3.read("""{"type":"content_block_start","content_block":{"type":"text","text":"Hello"}}""")
    )
    @test isnothing(extract_content(AnthropicStream(), chunk_wrong_type))

    # Test case 3: Basic text_delta extraction
    chunk_text = StreamChunk(
        :content_block_delta,
        """{"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello world"}}""",
        JSON3.read("""{"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello world"}}""")
    )
    @test extract_content(AnthropicStream(), chunk_text) == "Hello world"

    # Test case 4: thinking_delta extraction with include_thinking=true (default)
    chunk_thinking = StreamChunk(
        :content_block_delta,
        """{"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"This is a thought"}}""",
        JSON3.read("""{"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"This is a thought"}}""")
    )
    @test extract_content(AnthropicStream(), chunk_thinking) == "This is a thought"

    # Test case 5: thinking_delta with include_thinking=false should return nothing
    @test isnothing(extract_content(
        AnthropicStream(), chunk_thinking, include_thinking = false))

    # Test case 6: Content block delta with unexpected delta type
    chunk_unknown = StreamChunk(
        :content_block_delta,
        """{"type":"content_block_delta","delta":{"type":"unknown_delta","content":"Something"}}""",
        JSON3.read("""{"type":"content_block_delta","delta":{"type":"unknown_delta","content":"Something"}}""")
    )
    @test isnothing(extract_content(AnthropicStream(), chunk_unknown))

    # Test case 7: Missing text field in text_delta should return nothing
    chunk_missing_text = StreamChunk(
        :content_block_delta,
        """{"type":"content_block_delta","delta":{"type":"text_delta"}}""",
        JSON3.read("""{"type":"content_block_delta","delta":{"type":"text_delta"}}""")
    )
    @test isnothing(extract_content(AnthropicStream(), chunk_missing_text))

    # Test case 8: Missing thinking field in thinking_delta should return nothing
    chunk_missing_thinking = StreamChunk(
        :content_block_delta,
        """{"type":"content_block_delta","delta":{"type":"thinking_delta"}}""",
        JSON3.read("""{"type":"content_block_delta","delta":{"type":"thinking_delta"}}""")
    )
    @test isnothing(extract_content(AnthropicStream(), chunk_missing_thinking))
end
