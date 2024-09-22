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
