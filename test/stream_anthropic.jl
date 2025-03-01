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

    # Test case 2: message_start event should return nothing
    chunk_message_start = StreamChunk(
        :message_start,
        """{"type":"message_start","message":{"id":"msg_01...","role":"assistant","content":[]}}""",
        JSON3.read("""{"type":"message_start","message":{"id":"msg_01...","role":"assistant","content":[]}}""")
    )
    @test isnothing(extract_content(AnthropicStream(), chunk_message_start))

    # Test case 3: content_block_start with text should return the text
    chunk_block_start = StreamChunk(
        :content_block_start,
        """{"type":"content_block_start","index":0,"content_block":{"type":"text","text":"Hi!"}}""",
        JSON3.read("""{"type":"content_block_start","index":0,"content_block":{"type":"text","text":"Hi!"}}""")
    )
    @test extract_content(AnthropicStream(), chunk_block_start) == "Hi!"

    # Test case 4: content_block_stop without content_block should return nothing
    chunk_block_stop = StreamChunk(
        :content_block_stop,
        """{"type":"content_block_stop","index":0}""",
        JSON3.read("""{"type":"content_block_stop","index":0}""")
    )
    @test isnothing(extract_content(AnthropicStream(), chunk_block_stop))

    # Test case 5: text_delta extraction from content_block_delta
    chunk_text_delta = StreamChunk(
        :content_block_delta,
        """{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Hello world"}}""",
        JSON3.read("""{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Hello world"}}""")
    )
    @test extract_content(AnthropicStream(), chunk_text_delta) == "Hello world"

    # Test case 6: thinking_delta extraction with include_thinking=true (default)
    chunk_thinking_delta = StreamChunk(
        :content_block_delta,
        """{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"This is a thought"}}""",
        JSON3.read("""{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"This is a thought"}}""")
    )
    @test extract_content(AnthropicStream(), chunk_thinking_delta) == "This is a thought"

    # Test case 7: thinking_delta with include_thinking=false should return nothing
    @test isnothing(extract_content(
        AnthropicStream(), chunk_thinking_delta, include_thinking = false))

    # Test case 8: signature_delta should return nothing
    chunk_signature_delta = StreamChunk(
        :content_block_delta,
        """{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"EqQBCgIYAhIM1gbcDa..."}}""",
        JSON3.read("""{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"EqQBCgIYAhIM1gbcDa..."}}""")
    )
    @test isnothing(extract_content(AnthropicStream(), chunk_signature_delta))

    # Test case 9: message_delta should return nothing
    chunk_message_delta = StreamChunk(
        :message_delta,
        """{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null}}""",
        JSON3.read("""{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null}}""")
    )
    @test isnothing(extract_content(AnthropicStream(), chunk_message_delta))

    # Test case 10: message_stop should return nothing
    chunk_message_stop = StreamChunk(
        :message_stop,
        """{"type":"message_stop"}""",
        JSON3.read("""{"type":"message_stop"}""")
    )
    @test isnothing(extract_content(AnthropicStream(), chunk_message_stop))

    # Test case 11: Empty text in text_delta should return empty string
    chunk_empty_text = StreamChunk(
        :content_block_delta,
        """{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":""}}""",
        JSON3.read("""{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":""}}""")
    )
    @test extract_content(AnthropicStream(), chunk_empty_text) == ""

    # Test case 12: Missing text field in text_delta should return nothing
    chunk_missing_text = StreamChunk(
        :content_block_delta,
        """{"type":"content_block_delta","index":1,"delta":{"type":"text_delta"}}""",
        JSON3.read("""{"type":"content_block_delta","index":1,"delta":{"type":"text_delta"}}""")
    )
    @test isnothing(extract_content(AnthropicStream(), chunk_missing_text))

    # Test case 13: Missing thinking field in thinking_delta should return nothing
    chunk_missing_thinking = StreamChunk(
        :content_block_delta,
        """{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta"}}""",
        JSON3.read("""{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta"}}""")
    )
    @test isnothing(extract_content(AnthropicStream(), chunk_missing_thinking))

    # Test case 14: content_block_start with text should extract the text
    chunk_block_start_text = StreamChunk(
        :content_block_start,
        """{"type":"content_block_start","index":0,"content_block":{"type":"text","text":"Initial text"}}""",
        JSON3.read("""{"type":"content_block_start","index":0,"content_block":{"type":"text","text":"Initial text"}}""")
    )
    @test extract_content(AnthropicStream(), chunk_block_start_text) == "Initial text"

    # Test case 15: content_block_start with thinking should extract the thinking
    chunk_block_start_thinking = StreamChunk(
        :content_block_start,
        """{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"Initial thinking"}}""",
        JSON3.read("""{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"Initial thinking"}}""")
    )
    @test extract_content(AnthropicStream(), chunk_block_start_thinking) ==
          "Initial thinking"

    # Test case 16: content_block_start with thinking but include_thinking=false should return nothing
    @test isnothing(extract_content(
        AnthropicStream(), chunk_block_start_thinking, include_thinking = false))

    # Test case 17: content_block_stop with text should extract the text
    chunk_block_stop_text = StreamChunk(
        :content_block_stop,
        """{"type":"content_block_stop","index":0,"content_block":{"type":"text","text":"Final text"}}""",
        JSON3.read("""{"type":"content_block_stop","index":0,"content_block":{"type":"text","text":"Final text"}}""")
    )
    @test extract_content(AnthropicStream(), chunk_block_stop_text) == "Final text"

    # Test case 18: content_block_stop with thinking should extract the thinking
    chunk_block_stop_thinking = StreamChunk(
        :content_block_stop,
        """{"type":"content_block_stop","index":0,"content_block":{"type":"thinking","thinking":"Final thinking"}}""",
        JSON3.read("""{"type":"content_block_stop","index":0,"content_block":{"type":"thinking","thinking":"Final thinking"}}""")
    )
    @test extract_content(AnthropicStream(), chunk_block_stop_thinking) == "Final thinking"

    # Test case 19: content_block_stop with thinking but include_thinking=false should return nothing
    @test isnothing(extract_content(
        AnthropicStream(), chunk_block_stop_thinking, include_thinking = false))
end
