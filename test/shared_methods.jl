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
    @test_logs (:info, r"Incomplete message detected") chunks, spillover=extract_chunks(
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
    chunks, spillover = extract_chunks(
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
