@testset "is_done-ResponseStream" begin
    # Test case 1: response.completed event should return true
    completed_chunk = StreamChunk(
        :response_completed,
        """{"type":"response.completed","response":{"id":"resp_123"}}""",
        JSON3.read("""{"type":"response.completed","response":{"id":"resp_123"}}""")
    )
    @test is_done(ResponseStream(), completed_chunk) == true

    # Test case 2: other events should return false
    other_chunk = StreamChunk(
        :response_output_text_delta,
        """{"type":"response.output_text.delta","delta":"Hello"}""",
        JSON3.read("""{"type":"response.output_text.delta","delta":"Hello"}""")
    )
    @test is_done(ResponseStream(), other_chunk) == false

    # Test case 3: non-JSON chunk should return false
    non_json_chunk = StreamChunk(
        :data,
        "Plain text",
        nothing
    )
    @test is_done(ResponseStream(), non_json_chunk) == false

    # Test case 4: JSON without type field should return false
    no_type_chunk = StreamChunk(
        :data,
        """{"other":"field"}""",
        JSON3.read("""{"other":"field"}""")
    )
    @test is_done(ResponseStream(), no_type_chunk) == false
end

@testset "extract_content-ResponseStream" begin
    # Test case 1: response.output_text.delta should return delta content
    output_delta_chunk = StreamChunk(
        :response_output_text_delta,
        """{"type":"response.output_text.delta","delta":"Hello world"}""",
        JSON3.read("""{"type":"response.output_text.delta","delta":"Hello world"}""")
    )
    @test extract_content(ResponseStream(), output_delta_chunk) == "Hello world"

    # Test case 2: response.reasoning_summary_text.delta should return italic formatted content
    reasoning_delta_chunk = StreamChunk(
        :response_reasoning_summary_text_delta,
        """{"type":"response.reasoning_summary_text.delta","delta":"This is reasoning"}""",
        JSON3.read("""{"type":"response.reasoning_summary_text.delta","delta":"This is reasoning"}""")
    )
    @test extract_content(ResponseStream(), reasoning_delta_chunk) == "\e[3mThis is reasoning\e[23m"

    # Test case 3: response.reasoning_summary_text.done should return newline
    reasoning_done_chunk = StreamChunk(
        :response_reasoning_summary_text_done,
        """{"type":"response.reasoning_summary_text.done"}""",
        JSON3.read("""{"type":"response.reasoning_summary_text.done"}""")
    )
    @test extract_content(ResponseStream(), reasoning_done_chunk) == "\n"

    # Test case 4: other event types should return nothing
    other_chunk = StreamChunk(
        :response_created,
        """{"type":"response.created","response":{"id":"resp_123"}}""",
        JSON3.read("""{"type":"response.created","response":{"id":"resp_123"}}""")
    )
    @test isnothing(extract_content(ResponseStream(), other_chunk))

    # Test case 5: non-JSON chunk should return nothing
    non_json_chunk = StreamChunk(
        :data,
        "Plain text",
        nothing
    )
    @test isnothing(extract_content(ResponseStream(), non_json_chunk))

    # Test case 6: JSON without delta field should return nothing
    no_delta_chunk = StreamChunk(
        :response_output_text_delta,
        """{"type":"response.output_text.delta"}""",
        JSON3.read("""{"type":"response.output_text.delta"}""")
    )
    @test isnothing(extract_content(ResponseStream(), no_delta_chunk))

    # Test case 7: empty delta should return nothing
    empty_delta_chunk = StreamChunk(
        :response_output_text_delta,
        """{"type":"response.output_text.delta","delta":""}""",
        JSON3.read("""{"type":"response.output_text.delta","delta":""}""")
    )
    @test extract_content(ResponseStream(), empty_delta_chunk) == ""

    # Test case 8: reasoning delta with nothing should return nothing
    reasoning_empty_chunk = StreamChunk(
        :response_reasoning_summary_text_delta,
        """{"type":"response.reasoning_summary_text.delta","delta":null}""",
        JSON3.read("""{"type":"response.reasoning_summary_text.delta","delta":null}""")
    )
    @test isnothing(extract_content(ResponseStream(), reasoning_empty_chunk))
end

@testset "build_response_body-ResponseStream" begin
    # Test case 1: Empty chunks
    cb_empty = StreamCallback(flavor = ResponseStream())
    response = build_response_body(ResponseStream(), cb_empty)
    @test isnothing(response)

    # Test case 2: Simple response with output text deltas
    cb_simple = StreamCallback(flavor = ResponseStream())
    push!(cb_simple.chunks,
        StreamChunk(
            :response_created,
            """{"type":"response.created","response":{"id":"resp_123","status":"in_progress","output":[]}}""",
            JSON3.read("""{"type":"response.created","response":{"id":"resp_123","status":"in_progress","output":[]}}""")
        ))
    push!(cb_simple.chunks,
        StreamChunk(
            :response_output_text_delta,
            """{"type":"response.output_text.delta","delta":"Hello"}""",
            JSON3.read("""{"type":"response.output_text.delta","delta":"Hello"}""")
        ))
    push!(cb_simple.chunks,
        StreamChunk(
            :response_output_text_delta,
            """{"type":"response.output_text.delta","delta":" world"}""",
            JSON3.read("""{"type":"response.output_text.delta","delta":" world"}""")
        ))
    push!(cb_simple.chunks,
        StreamChunk(
            :response_completed,
            """{"type":"response.completed","response":{"id":"resp_123","status":"completed","output":[{"type":"message","status":"completed","content":[{"type":"output_text","text":""}],"role":"assistant"}]}}""",
            JSON3.read("""{"type":"response.completed","response":{"id":"resp_123","status":"completed","output":[{"type":"message","status":"completed","content":[{"type":"output_text","text":""}],"role":"assistant"}]}}""")
        ))
    response = build_response_body(ResponseStream(), cb_simple)
    @test response[:id] == "resp_123"
    @test response[:status] == "completed"
    @test response[:output][1][:content][1][:text] == "Hello world"

    # Test case 3: Response without initial response.created event
    cb_no_created = StreamCallback(flavor = ResponseStream())
    push!(cb_no_created.chunks,
        StreamChunk(
            :response_output_text_delta,
            """{"type":"response.output_text.delta","delta":"Direct"}""",
            JSON3.read("""{"type":"response.output_text.delta","delta":"Direct"}""")
        ))
    push!(cb_no_created.chunks,
        StreamChunk(
            :response_completed,
            """{"type":"response.completed","response":{"id":"resp_456","status":"completed"}}""",
            JSON3.read("""{"type":"response.completed","response":{"id":"resp_456","status":"completed"}}""")
        ))
    response = build_response_body(ResponseStream(), cb_no_created)
    @test response[:id] == "resp_456"
    @test response[:output][1][:content][1][:text] == "Direct"

    # Test case 4: Response with existing output structure
    cb_existing_output = StreamCallback(flavor = ResponseStream())
    push!(cb_existing_output.chunks,
        StreamChunk(
            :response_created,
            """{"type":"response.created","response":{"id":"resp_789","output":[{"type":"message","content":[{"type":"output_text","text":"Initial"}],"role":"assistant"}]}}""",
            JSON3.read("""{"type":"response.created","response":{"id":"resp_789","output":[{"type":"message","content":[{"type":"output_text","text":"Initial"}],"role":"assistant"}]}}""")
        ))
    push!(cb_existing_output.chunks,
        StreamChunk(
            :response_output_text_delta,
            """{"type":"response.output_text.delta","delta":"Updated"}""",
            JSON3.read("""{"type":"response.output_text.delta","delta":"Updated"}""")
        ))
    push!(cb_existing_output.chunks,
        StreamChunk(
            :response_completed,
            """{"type":"response.completed","response":{"id":"resp_789","status":"completed"}}""",
            JSON3.read("""{"type":"response.completed","response":{"id":"resp_789","status":"completed"}}""")
        ))
    response = build_response_body(ResponseStream(), cb_existing_output)
    @test response[:id] == "resp_789"
    @test response[:output][1][:content][1][:text] == "Updated"

    # Test case 5: Multiple content deltas
    cb_multiple = StreamCallback(flavor = ResponseStream())
    push!(cb_multiple.chunks,
        StreamChunk(
            :response_output_text_delta,
            """{"type":"response.output_text.delta","delta":"Part "}""",
            JSON3.read("""{"type":"response.output_text.delta","delta":"Part "}""")
        ))
    push!(cb_multiple.chunks,
        StreamChunk(
            :response_output_text_delta,
            """{"type":"response.output_text.delta","delta":"one "}""",
            JSON3.read("""{"type":"response.output_text.delta","delta":"one "}""")
        ))
    push!(cb_multiple.chunks,
        StreamChunk(
            :response_output_text_delta,
            """{"type":"response.output_text.delta","delta":"and "}""",
            JSON3.read("""{"type":"response.output_text.delta","delta":"and "}""")
        ))
    push!(cb_multiple.chunks,
        StreamChunk(
            :response_output_text_delta,
            """{"type":"response.output_text.delta","delta":"two"}""",
            JSON3.read("""{"type":"response.output_text.delta","delta":"two"}""")
        ))
    push!(cb_multiple.chunks,
        StreamChunk(
            :response_completed,
            """{"type":"response.completed","response":{"id":"resp_multi","status":"completed"}}""",
            JSON3.read("""{"type":"response.completed","response":{"id":"resp_multi","status":"completed"}}""")
        ))
    response = build_response_body(ResponseStream(), cb_multiple)
    @test response[:output][1][:content][1][:text] == "Part one and two"

    # Test case 6: Response completed without content deltas - no output structure created
    cb_no_deltas = StreamCallback(flavor = ResponseStream())
    push!(cb_no_deltas.chunks,
        StreamChunk(
            :response_completed,
            """{"type":"response.completed","response":{"id":"resp_empty","status":"completed"}}""",
            JSON3.read("""{"type":"response.completed","response":{"id":"resp_empty","status":"completed"}}""")
        ))
    response = build_response_body(ResponseStream(), cb_no_deltas)
    @test response[:id] == "resp_empty"
    @test !haskey(response, :output)  # No content parts means no output structure created

    # Test case 8: Reasoning summary delta followed by done creates italic line with newline
    cb_reasoning = StreamCallback(flavor = ResponseStream())
    push!(cb_reasoning.chunks,
        StreamChunk(
            :response_reasoning_summary_text_delta,
            """{"type":"response.reasoning_summary_text.delta","delta":"Thinking..."}""",
            JSON3.read("""{"type":"response.reasoning_summary_text.delta","delta":"Thinking..."}""")
        ))
    push!(cb_reasoning.chunks,
        StreamChunk(
            :response_reasoning_summary_text_done,
            """{"type":"response.reasoning_summary_text.done"}""",
            JSON3.read("""{"type":"response.reasoning_summary_text.done"}""")
        ))
    # Test concatenated extract_content output
    reasoning_content = extract_content(ResponseStream(), cb_reasoning.chunks[1])
    done_content = extract_content(ResponseStream(), cb_reasoning.chunks[2])
    @test reasoning_content * done_content == "\e[3mThinking...\e[23m\n"

    # Test case 9: Response with error metadata
    cb_error = StreamCallback(flavor = ResponseStream())
    push!(cb_error.chunks,
        StreamChunk(
            :response_completed,
            """{"type":"response.completed","response":{"id":"resp_error","status":"error","error":{"type":"api_error","message":"Something went wrong"}}}""",
            JSON3.read("""{"type":"response.completed","response":{"id":"resp_error","status":"error","error":{"type":"api_error","message":"Something went wrong"}}}""")
        ))
    response = build_response_body(ResponseStream(), cb_error)
    @test response[:id] == "resp_error"
    @test response[:status] == "error"
    @test response[:error][:type] == "api_error"
    @test response[:error][:message] == "Something went wrong"

    # Test case 7: Only non-JSON chunks should return nothing
    cb_non_json = StreamCallback(flavor = ResponseStream())
    push!(cb_non_json.chunks,
        StreamChunk(
            :data,
            "Plain text",
            nothing
        ))
    response = build_response_body(ResponseStream(), cb_non_json)
    @test isnothing(response)
end