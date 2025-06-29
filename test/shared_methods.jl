@testset "is_done" begin
    # Test OpenAIStream
    openai_flavor = OpenAIStream()

    # Test when streaming is done
    done_chunk = StreamChunk(data = "[DONE]")
    @test is_done(openai_flavor, done_chunk) == true

    # Test when streaming is not done
    not_done_chunk = StreamChunk(data = "Some content")
    @test is_done(openai_flavor, not_done_chunk) == false

    # Test with empty data
    empty_chunk = StreamChunk(data = "")
    @test is_done(openai_flavor, empty_chunk) == false

    # Test AnthropicStream
    anthropic_flavor = AnthropicStream()

    # Test when streaming is done due to error
    error_chunk = StreamChunk(event = :error)
    @test is_done(anthropic_flavor, error_chunk) == true

    # Test when streaming is done due to message stop
    stop_chunk = StreamChunk(event = :message_stop)
    @test is_done(anthropic_flavor, stop_chunk) == true

    # Test when streaming is not done
    continue_chunk = StreamChunk(event = :content_block_start)
    @test is_done(anthropic_flavor, continue_chunk) == false

    # Test with nil event
    nil_event_chunk = StreamChunk(event = nothing)
    @test is_done(anthropic_flavor, nil_event_chunk) == false

    # Test with unsupported flavor
    struct UnsupportedFlavor <: AbstractStreamFlavor end
    unsupported_flavor = UnsupportedFlavor()
    @test_throws ArgumentError is_done(unsupported_flavor, StreamChunk())
end

@testset "extract_chunks" begin
    # Test basic functionality
    blob = "event: start\ndata: {\"key\": \"value\"}\n\nevent: end\ndata: {\"status\": \"complete\"}\n\n"
    chunks, spillover = extract_chunks(OpenAIStream(), blob)
    @test length(chunks) == 2
    @test chunks[1].event == :start
    @test chunks[1].json == JSON3.read("{\"key\": \"value\"}")
    @test chunks[2].event == :end
    @test chunks[2].json == JSON3.read("{\"status\": \"complete\"}")
    @test spillover == ""

    # Test with spillover - SSE spec compliant
    blob_with_spillover = "event: start\ndata: {\"key\": \"value\"}\n\nevent: continue\ndata: {\"partial\": \"data"
    @test_logs (:info, r"Incomplete message detected") chunks,
    spillover=extract_chunks(
        OpenAIStream(), blob_with_spillover; verbose = true)
    chunks, spillover = extract_chunks(
        OpenAIStream(), blob_with_spillover; verbose = true)
    @test length(chunks) == 1
    @test chunks[1].event == :start
    @test chunks[1].json == JSON3.read("{\"key\": \"value\"}")
    @test spillover == "event: continue\ndata: {\"partial\": \"data"

    # Test with incoming spillover
    incoming_spillover = spillover
    blob_after_spillover = "\"}\n\nevent: end\ndata: {\"status\": \"complete\"}\n\n"
    chunks,
    spillover = extract_chunks(
        OpenAIStream(), blob_after_spillover; spillover = incoming_spillover)
    @test length(chunks) == 2
    @test chunks[1].json == JSON3.read("{\"partial\": \"data\"}")
    @test chunks[2].event == :end
    @test chunks[2].json == JSON3.read("{\"status\": \"complete\"}")
    @test spillover == ""

    # Test with multiple data fields per event - SSE spec compliant (joined with newlines)
    multi_data_blob = "event: multi\ndata: line1\ndata: line2\n\n"
    chunks, spillover = extract_chunks(OpenAIStream(), multi_data_blob)
    @test length(chunks) == 1
    @test chunks[1].event == :multi
    @test chunks[1].data == "line1\nline2"

    # Test with non-JSON data
    non_json_blob = "event: text\ndata: This is plain text\n\n"
    chunks, spillover = extract_chunks(OpenAIStream(), non_json_blob)
    @test length(chunks) == 1
    @test chunks[1].event == :text
    @test chunks[1].data == "This is plain text"
    @test isnothing(chunks[1].json)

    # Test with empty blob
    empty_blob = ""
    chunks, spillover = extract_chunks(OpenAIStream(), empty_blob)
    @test isempty(chunks)
    @test spillover == ""

    # Test with malformed JSON
    malformed_json_blob = "event: error\ndata: {\"key\": \"value\",}\n\n"
    chunks, spillover = extract_chunks(
        OpenAIStream(), malformed_json_blob; verbose = true)
    @test length(chunks) == 1
    @test chunks[1].event == :error
    @test chunks[1].data == "{\"key\": \"value\",}"
    @test isnothing(chunks[1].json)

    # Test with multiple data fields, no event
    blob_no_event = "data: {\"key\": \"value\"}\n\ndata: {\"partial\": \"data\"}\n\ndata: {\"status\": \"complete\"}\n\n"
    chunks, spillover = extract_chunks(OpenAIStream(), blob_no_event)
    @test length(chunks) == 3
    @test chunks[1].data == "{\"key\": \"value\"}"
    @test chunks[2].data == "{\"partial\": \"data\"}"
    @test chunks[3].data == "{\"status\": \"complete\"}"
    @test spillover == ""

    # Test case for s1: Multiple events and data chunks
    s1 = """event: test
    data: {"id":"chatcmpl-A3zvq9GWhji7h1Gz0gKNIn9r2tABJ","object":"chat.completion.chunk","created":1725516414,"model":"gpt-4o-mini-2024-07-18","system_fingerprint":"fp_f905cf32a9","choices":[{"index":0,"delta":{"content":","},"logprobs":null,"finish_reason":null}]}

    event: test2
    data: {"id":"chatcmpl-A3zvq9GWhji7h1Gz0gKNIn9r2tABJ","object":"chat.completion.chunk","created":1725516414,"model":"gpt-4o-mini-2024-07-18","system_fingerprint":"fp_f905cf32a9","choices":[{"index":0,"delta":{"content":" "},"logprobs":null,"finish_reason":null}]}

    data: [DONE]

    """
    chunks, spillover = extract_chunks(OpenAIStream(), s1)
    @test length(chunks) == 3
    @test chunks[1].event == :test
    @test chunks[2].event == :test2
    @test chunks[3].data == "[DONE]"
    @test spillover == ""

    @test extract_content(OpenAIStream(), chunks[1]) == ","
    @test extract_content(OpenAIStream(), chunks[2]) == " "

    # Test case for s2: Multiple data chunks without events
    s2 = """data: {"id":"chatcmpl-A3zvq9GWhji7h1Gz0gKNIn9r2tABJ","object":"chat.completion.chunk","created":1725516414,"model":"gpt-4o-mini-2024-07-18","system_fingerprint":"fp_f905cf32a9","choices":[{"index":0,"delta":{"content":","},"logprobs":null,"finish_reason":null}]}

      data: {"id":"chatcmpl-A3zvq9GWhji7h1Gz0gKNIn9r2tABJ","object":"chat.completion.chunk","created":1725516414,"model":"gpt-4o-mini-2024-07-18","system_fingerprint":"fp_f905cf32a9","choices":[{"index":0,"delta":{"content":" "},"logprobs":null,"finish_reason":null}]}

      data: [DONE]

      """
    chunks, spillover = extract_chunks(OpenAIStream(), s2)
    @test length(chunks) == 3
    @test all(chunk.event === nothing for chunk in chunks)
    @test chunks[3].data == "[DONE]"
    @test spillover == ""

    # Test case for s3: Simple data chunks - SSE spec compliant (joined with newlines)
    s3 = """data: a
    data: b
    data: c

    data: [DONE]

    """
    chunks, spillover = extract_chunks(OpenAIStream(), s3)
    @test length(chunks) == 2
    @test chunks[1].data == "a\nb\nc"
    @test chunks[2].data == "[DONE]"
    @test spillover == ""

    # Test case for s4a and s4b: Handling spillover
    s4a = """event: test
    data: {"id":"chatcmpl-A3zvq9GWhji7h1Gz0gKNIn9r2tABJ","object":"chat.completion.chunk","created":1725516414,"model":"gpt-4o-mini-2024-07-18","system_fingerprint":"fp_f905cf32a9","choices":[{"index":0,"delta":{"content":","},"logprobs":null,"finish_reason":null}]}

    event: test2
    data: {"id":"chatcmpl-A3zvq9GWhji7h1Gz0gKNIn9r2tABJ","object":"chat.completion.chunk","created"""
    s4b = """":1725516414,"model":"gpt-4o-mini-2024-07-18","system_fingerprint":"fp_f905cf32a9","choices":[{"index":0,"delta":{"content":" "},"logprobs":null,"finish_reason":null}]}

    data: [DONE]

    """
    chunks, spillover = extract_chunks(OpenAIStream(), s4a)
    @test length(chunks) == 1
    @test chunks[1].event == :test
    @test !isempty(spillover)

    chunks, final_spillover = extract_chunks(
        OpenAIStream(), s4b; spillover = spillover)
    @test length(chunks) == 2
    @test chunks[2].data == "[DONE]"
    @test final_spillover == ""

    # Test with real Anthropic LLM response streams captured from test_data_clip_issue.jl
    # This tests SSE spec compliance with actual data that includes "data:" patterns in the content
    real_anthropic_blob = """event: message_start
data: {"type":"message_start","message":{"id":"msg_01Kf4tf7utCTCiPTBYteMCUS","type":"message","role":"assistant","model":"claude-3-5-haiku-20241022","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":32,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":2,"service_tier":"standard"}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: ping
data: {"type": "ping"}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Here you"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" go:\\n\\ndata"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":": [1], data"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":": [2], data"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":": [3], data"}}

event: message_stop
data: {"type":"message_stop"}

"""
    chunks, spillover = extract_chunks(AnthropicStream(), real_anthropic_blob)
    @test length(chunks) == 9
    @test spillover == ""

    # Test that SSE spec compliance correctly handles "data:" patterns in content
    content_deltas = filter(chunk -> chunk.event == :content_block_delta, chunks)
    @test length(content_deltas) == 5

    # Verify that raw data contains "data" patterns exactly as sent by LLM  
    # Note: Looking for "data" in JSON content, not the SSE "data:" field
    data_containing_chunks = filter(
        chunk -> contains(chunk.data, "\"text\":\"") && contains(chunk.data, "data"),
        content_deltas)
    @test length(data_containing_chunks) == 4  # chunks 5,6,7,8 contain "data" in the text field

    # Test specific content extraction
    @test chunks[1].event == :message_start
    @test chunks[2].event == :content_block_start
    @test chunks[3].event == :ping
    @test chunks[9].event == :message_stop

    # Test that content contains the exact "data:" patterns as generated by LLM
    text_delta_chunk = chunks[6] # chunk with ": [1], data"
    @test text_delta_chunk.event == :content_block_delta
    @test contains(text_delta_chunk.data, "data")
    parsed_json = text_delta_chunk.json
    @test parsed_json.delta.text == ": [1], data"
end

@testset "SSE spec compliance fixes" begin
    # Test 1: BOM handling - UTF-8 BOM should be stripped from field names
    bom_blob = "\ufeffdata: message with BOM\nevent: test_event\n\n"
    chunks, spillover = extract_chunks(OpenAIStream(), bom_blob)
    @test length(chunks) == 1
    @test chunks[1].data == "message with BOM"
    @test chunks[1].event == :test_event
    @test spillover == ""

    # Test 2: BOM in field value should be preserved
    bom_value_blob = "data: \ufeffmessage with BOM in value\n\n"
    chunks, spillover = extract_chunks(OpenAIStream(), bom_value_blob)
    @test length(chunks) == 1
    @test chunks[1].data == "\ufeffmessage with BOM in value"

    # Test 3: Empty data fields should create proper empty strings (no artifacts)
    empty_data_blob = "data:\ndata: \ndata:  \nevent: test\n\n"
    chunks, spillover = extract_chunks(OpenAIStream(), empty_data_blob)
    @test length(chunks) == 1
    @test chunks[1].data == "\n\n "  # Three data fields: empty, space, two spaces
    @test chunks[1].event == :test

    # Test 4: Multiple empty data fields
    multi_empty_blob = "data:\ndata:\ndata:\n\n"
    chunks, spillover = extract_chunks(OpenAIStream(), multi_empty_blob)
    @test length(chunks) == 1
    @test chunks[1].data == "\n\n"  # Three empty data fields joined with newlines

    # Test 5: Empty event field should be ignored
    empty_event_blob = "data: test message\nevent:\nevent: \n\n"
    chunks, spillover = extract_chunks(OpenAIStream(), empty_event_blob)
    @test length(chunks) == 1
    @test chunks[1].data == "test message"
    @test chunks[1].event === nothing  # Empty event fields should be ignored

    # Test 6: Error handling - malformed lines should be handled gracefully
    # The error handling prevents crashes but may not always generate warnings
    malformed_blob = "data: valid message\n\x00invalid: line with null\ndata: another valid\n\n"
    chunks, spillover = extract_chunks(OpenAIStream(), malformed_blob; verbose = false)
    @test length(chunks) == 1
    @test chunks[1].data == "valid message\nanother valid"

    # Test 7: Invalid UTF-8 sequences should be handled gracefully
    invalid_utf8_blob = "data: valid\n\xff\xfe: invalid utf8\ndata: also valid\n\n"
    chunks, spillover = extract_chunks(OpenAIStream(), invalid_utf8_blob; verbose = false)  # No warnings for silent test
    @test length(chunks) == 1
    @test chunks[1].data == "valid\nalso valid"

    # Test 8: Unicode field names with BOM
    unicode_bom_blob = "\ufeff测试: unicode field\ndata: test data\n\n"
    chunks, spillover = extract_chunks(OpenAIStream(), unicode_bom_blob)
    @test length(chunks) == 1
    @test chunks[1].data == "test data"  # Unicode field name should be ignored (not 'data')

    # Test 9: Mixed valid/invalid lines
    mixed_blob = """data: line1
    invalid line without colon
    event: test_event
    data: line2
    : this is a comment
    data: line3

    """
    chunks, spillover = extract_chunks(OpenAIStream(), mixed_blob)
    @test length(chunks) == 1
    @test chunks[1].data == "line1\nline2\nline3"
    @test chunks[1].event == :test_event

    # Test 10: Edge case - field name that becomes empty after BOM removal
    edge_bom_blob = "\ufeff: value after empty field\ndata: valid data\n\n"
    chunks, spillover = extract_chunks(OpenAIStream(), edge_bom_blob)
    @test length(chunks) == 1
    @test chunks[1].data == "valid data"  # BOM+colon line should be treated as comment
end

@testset "print_content" begin
    # Test printing to IO
    io = IOBuffer()
    print_content(io, "Test content")
    @test String(take!(io)) == "Test content"

    # Test printing to Channel
    ch = Channel{String}(1)
    print_content(ch, "Channel content")
    @test take!(ch) == "Channel content"

    # Test printing to nothing
    @test print_content(nothing, "No output") === nothing
end

@testset "callback" begin
    # Test with valid content
    io = IOBuffer()
    cb = StreamCallback(out = io, flavor = OpenAIStream())
    valid_chunk = StreamChunk(
        nothing,
        """{"choices":[{"delta":{"content":"Hello"}}]}""",
        JSON3.read("""{"choices":[{"delta":{"content":"Hello"}}]}""")
    )
    callback(cb, valid_chunk)
    @test String(take!(io)) == "Hello"

    # Test with no content
    io = IOBuffer()
    cb = StreamCallback(out = io, flavor = OpenAIStream())
    no_content_chunk = StreamChunk(
        nothing,
        """{"choices":[{"delta":{}}]}""",
        JSON3.read("""{"choices":[{"delta":{}}]}""")
    )
    callback(cb, no_content_chunk)
    @test isempty(take!(io))

    # Test with Channel output
    ch = Channel{String}(1)
    cb = StreamCallback(out = ch, flavor = OpenAIStream())
    callback(cb, valid_chunk)
    @test take!(ch) == "Hello"

    # Test with nothing output
    cb = StreamCallback(out = nothing, flavor = OpenAIStream())
    @test callback(cb, valid_chunk) === nothing
end

@testset "handle_error_message" begin
    # Test case 1: No error
    chunk = StreamChunk(:content, "Normal content", nothing)
    @test isnothing(handle_error_message(chunk))

    # Test case 2: Error event
    error_chunk = StreamChunk(:error, "Error occurred", nothing)
    @test_logs (:warn, "Error detected in the streaming response: Error occurred") handle_error_message(error_chunk)

    # Test case 4: Detailed error in JSON
    obj = Dict(:error => Dict(:message => "Invalid input", :type => "user_error"))
    detailed_error_chunk = StreamChunk(
        nothing, JSON3.write(obj), JSON3.read(JSON3.write(obj)))
    @test_logs (:warn,
        r"Message: Invalid input") handle_error_message(detailed_error_chunk)
    @test_logs (:warn,
        r"Type: user_error") handle_error_message(detailed_error_chunk)

    # Test case 5: Throw on error
    @test_throws Exception handle_error_message(error_chunk, throw_on_error = true)
end

@testset "extract_content" begin
    # Test unimplemented flavor throws error
    struct TestFlavor <: AbstractStreamFlavor end
    test_chunk = StreamChunk(nothing, "test data", nothing)
    @test_throws ArgumentError extract_content(TestFlavor(), test_chunk)
end

## Not working yet!!
# @testset "streamed_request!" begin
#     # Setup mock server
#     PORT = rand(10000:20000)
#     server = HTTP.serve!(PORT; verbose = false) do request
#         if request.method == "POST" && request.target == "/v1/chat/completions"
#             # Simulate streaming response
#             return HTTP.Response() do io
#                 write(io, "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n")
#                 write(io, "data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n")
#                 write(io, "data: [DONE]\n\n")
#             end
#         else
#             return HTTP.Response(404, "Not found")
#         end
#     end

#     # Test streamed_request!
#     url = "http://localhost:$PORT/v1/chat/completions"
#     headers = ["Content-Type" => "application/json"]
#     input = IOBuffer(JSON3.write(Dict(
#         "model" => "gpt-3.5-turbo",
#         "messages" => [Dict("role" => "user", "content" => "Say hello")]
#     )))

#     cb = StreamCallback(flavor = OpenAIStream())
#     response = streamed_request!(cb, url, headers, input)

#     # Assertions
#     @test response.status == 200
#     @test length(cb.chunks) == 3
#     @test cb.chunks[1].json.choices[1].delta.content == "Hello"
#     @test cb.chunks[2].json.choices[1].delta.content == " world"
#     @test cb.chunks[3].data == "[DONE]"

#     # Test build_response_body
#     body = build_response_body(OpenAIStream(), cb)
#     @test body[:choices][1][:message][:content] == "Hello world"
#     # Cleanup
#     close(server)
# end
