# Tests for OpenAI Responses API streaming

# Helper to load fixture and parse into chunks
function load_responses_fixture(filename)
    filepath = joinpath(@__DIR__, "fixtures", filename)
    content = read(filepath, String)
    chunks = StreamChunk[]

    for block in split(content, "\n\n")
        isempty(strip(block)) && continue
        event_name = nothing
        data_content = ""

        for line in split(block, '\n')
            line = rstrip(line, '\r')
            if startswith(line, "event: ")
                event_name = Symbol(strip(line[8:end]))
            elseif startswith(line, "data: ")
                data_content = strip(line[7:end])
            end
        end

        if !isempty(data_content)
            json = try
                JSON3.read(data_content)
            catch
                nothing
            end
            push!(chunks, StreamChunk(event_name, data_content, json))
        end
    end
    return chunks
end

@testset "OpenAIResponsesStream-is_done" begin
    flavor = OpenAIResponsesStream()

    # Should be done on response.completed
    completed_chunk = StreamChunk(
        Symbol("response.completed"),
        """{"type":"response.completed","sequence_number":15,"response":{"id":"resp_xxx","status":"completed"}}""",
        JSON3.read("""{"type":"response.completed","sequence_number":15,"response":{"id":"resp_xxx","status":"completed"}}""")
    )
    @test is_done(flavor, completed_chunk) == true

    # Should be done on response.failed
    failed_chunk = StreamChunk(
        Symbol("response.failed"),
        """{"type":"response.failed","error":{"message":"test error"}}""",
        JSON3.read("""{"type":"response.failed","error":{"message":"test error"}}""")
    )
    @test is_done(flavor, failed_chunk) == true

    # Should be done on response.incomplete
    incomplete_chunk = StreamChunk(
        Symbol("response.incomplete"),
        """{"type":"response.incomplete","response":{}}""",
        JSON3.read("""{"type":"response.incomplete","response":{}}""")
    )
    @test is_done(flavor, incomplete_chunk) == true

    # Should be done on error event
    error_chunk = StreamChunk(
        :error,
        """{"error":{"message":"Stream error"}}""",
        JSON3.read("""{"error":{"message":"Stream error"}}""")
    )
    @test is_done(flavor, error_chunk) == true

    # Should NOT be done on delta events
    delta_chunk = StreamChunk(
        Symbol("response.output_text.delta"),
        """{"type":"response.output_text.delta","sequence_number":4,"delta":"Hello"}""",
        JSON3.read("""{"type":"response.output_text.delta","sequence_number":4,"delta":"Hello"}""")
    )
    @test is_done(flavor, delta_chunk) == false

    # Should NOT be done on lifecycle events
    created_chunk = StreamChunk(
        Symbol("response.created"),
        """{"type":"response.created","sequence_number":0,"response":{"id":"resp_xxx"}}""",
        JSON3.read("""{"type":"response.created","sequence_number":0,"response":{"id":"resp_xxx"}}""")
    )
    @test is_done(flavor, created_chunk) == false

    in_progress_chunk = StreamChunk(
        Symbol("response.in_progress"),
        """{"type":"response.in_progress","sequence_number":1}""",
        JSON3.read("""{"type":"response.in_progress","sequence_number":1}""")
    )
    @test is_done(flavor, in_progress_chunk) == false

    # Should NOT be done on output_item events
    item_added_chunk = StreamChunk(
        Symbol("response.output_item.added"),
        """{"type":"response.output_item.added","item":{"type":"message"}}""",
        JSON3.read("""{"type":"response.output_item.added","item":{"type":"message"}}""")
    )
    @test is_done(flavor, item_added_chunk) == false
end

@testset "OpenAIResponsesStream-extract_content" begin
    flavor = OpenAIResponsesStream()

    # Test text delta extraction
    text_chunk = StreamChunk(
        Symbol("response.output_text.delta"),
        """{"type":"response.output_text.delta","sequence_number":4,"item_id":"msg_xxx","output_index":0,"content_index":0,"delta":"Hello","logprobs":[],"obfuscation":"xxx"}""",
        JSON3.read("""{"type":"response.output_text.delta","sequence_number":4,"item_id":"msg_xxx","output_index":0,"content_index":0,"delta":"Hello","logprobs":[],"obfuscation":"xxx"}""")
    )
    @test extract_content(flavor, text_chunk) == "Hello"

    # Test with empty delta
    empty_delta_chunk = StreamChunk(
        Symbol("response.output_text.delta"),
        """{"type":"response.output_text.delta","delta":""}""",
        JSON3.read("""{"type":"response.output_text.delta","delta":""}""")
    )
    @test extract_content(flavor, empty_delta_chunk) == ""

    # Test reasoning summary delta extraction with include_reasoning=true
    reasoning_chunk = StreamChunk(
        Symbol("response.reasoning_summary_text.delta"),
        """{"type":"response.reasoning_summary_text.delta","sequence_number":10,"delta":"Solving the problem..."}""",
        JSON3.read("""{"type":"response.reasoning_summary_text.delta","sequence_number":10,"delta":"Solving the problem..."}""")
    )
    @test extract_content(flavor, reasoning_chunk; include_reasoning = true) ==
          "Solving the problem..."

    # Test reasoning summary delta with include_reasoning=false
    @test isnothing(extract_content(flavor, reasoning_chunk; include_reasoning = false))

    # Test full reasoning text delta
    full_reasoning_chunk = StreamChunk(
        Symbol("response.reasoning_text.delta"),
        """{"type":"response.reasoning_text.delta","delta":"Step 1: Consider the inputs..."}""",
        JSON3.read("""{"type":"response.reasoning_text.delta","delta":"Step 1: Consider the inputs..."}""")
    )
    @test extract_content(flavor, full_reasoning_chunk; include_reasoning = true) ==
          "Step 1: Consider the inputs..."
    @test isnothing(extract_content(flavor, full_reasoning_chunk; include_reasoning = false))

    # Test non-content events return nothing
    created_chunk = StreamChunk(
        Symbol("response.created"),
        """{"type":"response.created","response":{"id":"resp_xxx"}}""",
        JSON3.read("""{"type":"response.created","response":{"id":"resp_xxx"}}""")
    )
    @test isnothing(extract_content(flavor, created_chunk))

    completed_chunk = StreamChunk(
        Symbol("response.completed"),
        """{"type":"response.completed","response":{"id":"resp_xxx"}}""",
        JSON3.read("""{"type":"response.completed","response":{"id":"resp_xxx"}}""")
    )
    @test isnothing(extract_content(flavor, completed_chunk))

    # Test chunk with no json returns nothing
    no_json_chunk = StreamChunk(Symbol("response.output_text.delta"), "invalid json", nothing)
    @test isnothing(extract_content(flavor, no_json_chunk))

    # Test chunk with missing delta field
    no_delta_chunk = StreamChunk(
        Symbol("response.output_text.delta"),
        """{"type":"response.output_text.delta","sequence_number":4}""",
        JSON3.read("""{"type":"response.output_text.delta","sequence_number":4}""")
    )
    @test isnothing(extract_content(flavor, no_delta_chunk))
end

@testset "OpenAIResponsesStream-build_response_body" begin
    flavor = OpenAIResponsesStream()

    # Test empty chunks
    cb_empty = StreamCallback()
    @test isnothing(build_response_body(flavor, cb_empty))

    # Test with manually constructed chunks
    cb = StreamCallback()
    push!(cb.chunks,
        StreamChunk(
            Symbol("response.created"),
            """{"type":"response.created","response":{"id":"resp_123","model":"gpt-4o","status":"in_progress"}}""",
            JSON3.read("""{"type":"response.created","response":{"id":"resp_123","model":"gpt-4o","status":"in_progress"}}""")
        ))
    push!(cb.chunks,
        StreamChunk(
            Symbol("response.output_text.delta"),
            """{"type":"response.output_text.delta","delta":"Hello"}""",
            JSON3.read("""{"type":"response.output_text.delta","delta":"Hello"}""")
        ))
    push!(cb.chunks,
        StreamChunk(
            Symbol("response.output_text.delta"),
            """{"type":"response.output_text.delta","delta":" world!"}""",
            JSON3.read("""{"type":"response.output_text.delta","delta":" world!"}""")
        ))
    push!(cb.chunks,
        StreamChunk(
            Symbol("response.completed"),
            """{"type":"response.completed","response":{"id":"resp_123","status":"completed","output":[],"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}""",
            JSON3.read("""{"type":"response.completed","response":{"id":"resp_123","status":"completed","output":[],"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}""")
        ))

    response = build_response_body(flavor, cb)
    @test !isnothing(response)
    @test response[:id] == "resp_123"
    @test response[:status] == "completed"
    @test haskey(response, :usage)
    @test response[:usage][:input_tokens] == 10
    @test response[:usage][:output_tokens] == 5
    @test response[:usage][:total_tokens] == 15

    # Test with reasoning content
    cb_reasoning = StreamCallback()
    push!(cb_reasoning.chunks,
        StreamChunk(
            Symbol("response.created"),
            """{"type":"response.created","response":{"id":"resp_456","model":"o4-mini"}}""",
            JSON3.read("""{"type":"response.created","response":{"id":"resp_456","model":"o4-mini"}}""")
        ))
    push!(cb_reasoning.chunks,
        StreamChunk(
            Symbol("response.reasoning_summary_text.delta"),
            """{"type":"response.reasoning_summary_text.delta","delta":"Thinking about "}""",
            JSON3.read("""{"type":"response.reasoning_summary_text.delta","delta":"Thinking about "}""")
        ))
    push!(cb_reasoning.chunks,
        StreamChunk(
            Symbol("response.reasoning_summary_text.delta"),
            """{"type":"response.reasoning_summary_text.delta","delta":"the problem..."}""",
            JSON3.read("""{"type":"response.reasoning_summary_text.delta","delta":"the problem..."}""")
        ))
    push!(cb_reasoning.chunks,
        StreamChunk(
            Symbol("response.output_text.delta"),
            """{"type":"response.output_text.delta","delta":"The answer is 4."}""",
            JSON3.read("""{"type":"response.output_text.delta","delta":"The answer is 4."}""")
        ))
    push!(cb_reasoning.chunks,
        StreamChunk(
            Symbol("response.completed"),
            """{"type":"response.completed","response":{"id":"resp_456","status":"completed","output":[],"usage":{"input_tokens":20,"output_tokens":10,"total_tokens":30}}}""",
            JSON3.read("""{"type":"response.completed","response":{"id":"resp_456","status":"completed","output":[],"usage":{"input_tokens":20,"output_tokens":10,"total_tokens":30}}}""")
        ))

    response_reasoning = build_response_body(flavor, cb_reasoning)
    @test !isnothing(response_reasoning)
    @test haskey(response_reasoning, :output)
    # Should have both reasoning and message in output
    @test length(response_reasoning[:output]) == 2
    reasoning_items = filter(o -> get(o, :type, "") == "reasoning", response_reasoning[:output])
    message_items = filter(o -> get(o, :type, "") == "message", response_reasoning[:output])
    @test length(reasoning_items) == 1
    @test length(message_items) == 1
end

@testset "OpenAIResponsesStream-fixture-simple" begin
    # Test with real fixture data (simple response)
    flavor = OpenAIResponsesStream()
    chunks = load_responses_fixture("responses_api_simple.txt")

    @test length(chunks) > 0

    # Verify we can identify completion
    completed_chunks = filter(c -> is_done(flavor, c), chunks)
    @test length(completed_chunks) == 1
    @test completed_chunks[1].event == Symbol("response.completed")

    # Build response and verify structure
    cb = StreamCallback()
    cb.chunks = chunks

    response = build_response_body(flavor, cb)
    @test !isnothing(response)
    @test haskey(response, :id)
    @test haskey(response, :model)
    @test haskey(response, :status)
    @test response[:status] == "completed"
    @test haskey(response, :usage)
    @test response[:usage][:input_tokens] > 0
    @test response[:usage][:output_tokens] > 0

    # Verify text was extracted correctly
    text_deltas = filter(c -> c.event == Symbol("response.output_text.delta"), chunks)
    @test length(text_deltas) > 0

    # Manually assemble text to compare
    assembled_text = join([get(c.json, :delta, "") for c in text_deltas], "")
    @test !isempty(assembled_text)
end

@testset "OpenAIResponsesStream-fixture-reasoning" begin
    # Test with real fixture data (reasoning response)
    flavor = OpenAIResponsesStream()
    chunks = load_responses_fixture("responses_api_reasoning.txt")

    @test length(chunks) > 0

    # Verify we have reasoning events
    reasoning_deltas = filter(
        c -> c.event == Symbol("response.reasoning_summary_text.delta"), chunks)
    @test length(reasoning_deltas) > 0

    # Verify we have text deltas
    text_deltas = filter(c -> c.event == Symbol("response.output_text.delta"), chunks)
    @test length(text_deltas) > 0

    # Build response
    cb = StreamCallback()
    cb.chunks = chunks

    response = build_response_body(flavor, cb)
    @test !isnothing(response)
    @test haskey(response, :output)

    # Should have output array from response.completed
    @test !isempty(response[:output])
end

@testset "OpenAIResponsesStream-callback-integration" begin
    # Test that callback() works correctly with OpenAIResponsesStream
    flavor = OpenAIResponsesStream()

    # Create a callback that writes to an IOBuffer
    output = IOBuffer()
    cb = StreamCallback(out = output, flavor = flavor)

    # Simulate receiving chunks
    chunks = [
        StreamChunk(
            Symbol("response.output_text.delta"),
            """{"delta":"Hello"}""",
            JSON3.read("""{"delta":"Hello"}""")
        ),
        StreamChunk(
            Symbol("response.output_text.delta"),
            """{"delta":" world"}""",
            JSON3.read("""{"delta":" world"}""")
        ),
        StreamChunk(
            Symbol("response.output_text.delta"),
            """{"delta":"!"}""",
            JSON3.read("""{"delta":"!"}""")
        )
    ]

    # Process each chunk through callback
    for chunk in chunks
        callback(cb, chunk)
        push!(cb, chunk)
    end

    # Verify output was written
    result = String(take!(output))
    @test result == "Hello world!"
end
