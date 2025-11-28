# Integration tests with SSE server
#
# These tests spin up a local HTTP server that replays fixture files
# as SSE streams to verify end-to-end parsing and response reconstruction.

using Sockets

# Helper to find an available port
function find_available_port()
    server = listen(0)
    port = getsockname(server)[2]
    close(server)
    return port
end

# Helper to read fixture file content
function read_fixture(filename)
    filepath = joinpath(@__DIR__, "fixtures", filename)
    return read(filepath, String)
end

# Create an SSE server that streams fixture content
function create_sse_server(port, fixture_content; delay_ms = 0)
    server = HTTP.listen!("127.0.0.1", port) do http
        # Read and discard the request body to avoid EOF errors
        try
            read(http)
        catch
        end

        # Set SSE headers
        HTTP.setstatus(http, 200)
        HTTP.setheader(http, "Content-Type" => "text/event-stream")
        HTTP.setheader(http, "Cache-Control" => "no-cache")
        HTTP.setheader(http, "Connection" => "keep-alive")
        HTTP.startwrite(http)

        # Stream the fixture content in chunks (split by double newlines)
        messages = split(fixture_content, "\n\n")
        for (i, message) in enumerate(messages)
            if !isempty(strip(message))
                # Write the message with double newline separator
                write(http, message * "\n\n")

                # Optional delay between messages to simulate real streaming
                if delay_ms > 0
                    sleep(delay_ms / 1000)
                end
            end
        end
    end
    return server
end

@testset "Integration-OpenAIResponsesStream" begin
    # Test with simple responses fixture
    @testset "simple-response" begin
        port = find_available_port()
        fixture_content = read_fixture("responses_api_simple.txt")

        server = create_sse_server(port, fixture_content)

        try
            sleep(0.1)

            cb = StreamCallback(out = nothing, flavor = OpenAIResponsesStream())
            url = "http://127.0.0.1:$port"
            headers = ["Content-Type" => "application/json"]
            payload = IOBuffer("""{"model":"gpt-4o-mini","input":"test","stream":true}""")

            resp = streamed_request!(cb, url, headers, payload)

            # Verify response
            @test resp.status == 200
            @test length(cb.chunks) > 0

            # Verify we received the expected events
            event_types = [c.event for c in cb.chunks if !isnothing(c.event)]
            @test Symbol("response.created") in event_types
            @test Symbol("response.output_text.delta") in event_types
            @test Symbol("response.completed") in event_types

            # Verify response body was reconstructed
            body = JSON3.read(resp.body)
            @test !isnothing(body)
            @test haskey(body, :id)
            @test haskey(body, :usage)
            @test body[:status] == "completed"
        finally
            close(server)
        end
    end

    # Test with reasoning fixture
    @testset "reasoning-response" begin
        port = find_available_port()
        fixture_content = read_fixture("responses_api_reasoning.txt")

        server = create_sse_server(port, fixture_content)

        try
            sleep(0.1)

            cb = StreamCallback(out = nothing, flavor = OpenAIResponsesStream())
            url = "http://127.0.0.1:$port"
            headers = ["Content-Type" => "application/json"]
            payload = IOBuffer("""{"model":"o4-mini","input":"test","stream":true}""")

            resp = streamed_request!(cb, url, headers, payload)

            @test resp.status == 200
            @test length(cb.chunks) > 0

            # Verify reasoning events were received
            reasoning_events = filter(
                c -> c.event == Symbol("response.reasoning_summary_text.delta"), cb.chunks)
            @test length(reasoning_events) > 0

            # Verify text events were received
            text_events = filter(
                c -> c.event == Symbol("response.output_text.delta"), cb.chunks)
            @test length(text_events) > 0

            # Verify response body
            body = JSON3.read(resp.body)
            @test !isnothing(body)
            @test haskey(body, :output)
            @test body[:status] == "completed"
        finally
            close(server)
        end
    end

    # Test streaming to IOBuffer (verifies content extraction)
    @testset "streaming-to-buffer" begin
        port = find_available_port()
        fixture_content = read_fixture("responses_api_simple.txt")

        server = create_sse_server(port, fixture_content)

        try
            sleep(0.1)

            # Stream to an IOBuffer to capture output
            output_buffer = IOBuffer()
            cb = StreamCallback(out = output_buffer, flavor = OpenAIResponsesStream())
            url = "http://127.0.0.1:$port"
            headers = ["Content-Type" => "application/json"]
            payload = IOBuffer("""{"model":"gpt-4o-mini","input":"test","stream":true}""")

            resp = streamed_request!(cb, url, headers, payload)

            # Get the streamed content
            streamed_text = String(take!(output_buffer))

            # Verify text was streamed
            @test !isempty(streamed_text)

            # The text should match what's in the fixture deltas
            # (We know from the fixture it says something like "2 + 2 equals 4.")
            @test occursin("2", streamed_text) || occursin("4", streamed_text)
        finally
            close(server)
        end
    end
end

@testset "Integration-OpenAIChatStream" begin
    # Test with real Chat Completions fixture from API
    @testset "real-fixture-response" begin
        port = find_available_port()
        fixture_content = read_fixture("chat_completions_simple.txt")

        server = create_sse_server(port, fixture_content)

        try
            sleep(0.1)

            output_buffer = IOBuffer()
            cb = StreamCallback(out = output_buffer, flavor = OpenAIChatStream())
            url = "http://127.0.0.1:$port"
            headers = ["Content-Type" => "application/json"]
            payload = IOBuffer("""{"model":"gpt-4o-mini","messages":[],"stream":true}""")

            resp = streamed_request!(cb, url, headers, payload)

            @test resp.status == 200
            @test length(cb.chunks) > 0

            # Verify streamed text (fixture contains "2 + 2 equals 4.")
            streamed_text = String(take!(output_buffer))
            @test occursin("2", streamed_text)
            @test occursin("4", streamed_text)

            # Verify response body reconstruction
            body = JSON3.read(resp.body)
            @test body[:object] == "chat.completion"
            @test haskey(body, :choices)
            @test length(body[:choices]) > 0
            @test body[:choices][1][:finish_reason] == "stop"
            @test haskey(body[:choices][1], :message)
            @test occursin("4", body[:choices][1][:message][:content])
        finally
            close(server)
        end
    end

    # Test with manually constructed fixture for specific behavior
    @testset "manual-fixture-response" begin
        port = find_available_port()

        # Manually create a Chat Completions style SSE stream (no event: prefix)
        chat_fixture = """
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13}}

data: [DONE]

"""

        server = create_sse_server(port, chat_fixture)

        try
            sleep(0.1)

            output_buffer = IOBuffer()
            cb = StreamCallback(out = output_buffer, flavor = OpenAIChatStream())
            url = "http://127.0.0.1:$port"
            headers = ["Content-Type" => "application/json"]
            payload = IOBuffer("""{"model":"gpt-4o-mini","messages":[],"stream":true}""")

            resp = streamed_request!(cb, url, headers, payload)

            @test resp.status == 200

            # Verify streamed text
            streamed_text = String(take!(output_buffer))
            @test streamed_text == "Hello world!"

            # Verify response body reconstruction
            body = JSON3.read(resp.body)
            @test body[:object] == "chat.completion"
            @test body[:choices][1][:message][:content] == "Hello world!"
            @test body[:choices][1][:finish_reason] == "stop"
        finally
            close(server)
        end
    end
end

@testset "Integration-ErrorHandling" begin
    # Test error event handling
    @testset "error-event" begin
        port = find_available_port()

        error_fixture = """
event: response.created
data: {"type":"response.created","response":{"id":"resp_err"}}

event: error
data: {"error":{"message":"Test error message","type":"server_error"}}

"""

        server = create_sse_server(port, error_fixture)

        try
            sleep(0.1)

            cb = StreamCallback(out = nothing, flavor = OpenAIResponsesStream(), throw_on_error = false)
            url = "http://127.0.0.1:$port"
            headers = ["Content-Type" => "application/json"]
            payload = IOBuffer("""{"model":"gpt-4o","input":"test","stream":true}""")

            # Should complete without throwing (throw_on_error = false)
            resp = streamed_request!(cb, url, headers, payload)

            @test resp.status == 200

            # Verify error event was captured
            error_chunks = filter(c -> c.event == :error, cb.chunks)
            @test length(error_chunks) == 1
        finally
            close(server)
        end
    end

    # Test response.failed handling
    @testset "failed-response" begin
        port = find_available_port()

        failed_fixture = """
event: response.created
data: {"type":"response.created","response":{"id":"resp_fail"}}

event: response.failed
data: {"type":"response.failed","response":{"id":"resp_fail","status":"failed","error":{"message":"Content policy violation"}}}

"""

        server = create_sse_server(port, failed_fixture)

        try
            sleep(0.1)

            cb = StreamCallback(out = nothing, flavor = OpenAIResponsesStream())
            url = "http://127.0.0.1:$port"
            headers = ["Content-Type" => "application/json"]
            payload = IOBuffer("""{"model":"gpt-4o","input":"test","stream":true}""")

            resp = streamed_request!(cb, url, headers, payload)

            @test resp.status == 200

            # Verify stream ended at response.failed
            @test any(c -> c.event == Symbol("response.failed"), cb.chunks)
        finally
            close(server)
        end
    end
end

@testset "Integration-ChunkedDelivery" begin
    # Test that chunked/partial message delivery is handled correctly
    @testset "partial-chunks" begin
        port = find_available_port()
        fixture_content = read_fixture("responses_api_simple.txt")

        # Use a small delay to simulate real streaming
        server = create_sse_server(port, fixture_content; delay_ms = 5)

        try
            sleep(0.1)

            cb = StreamCallback(out = nothing, flavor = OpenAIResponsesStream())
            url = "http://127.0.0.1:$port"
            headers = ["Content-Type" => "application/json"]
            payload = IOBuffer("""{"model":"gpt-4o-mini","input":"test","stream":true}""")

            resp = streamed_request!(cb, url, headers, payload)

            @test resp.status == 200
            @test length(cb.chunks) > 0

            # Verify all expected events were received despite chunked delivery
            @test any(c -> c.event == Symbol("response.created"), cb.chunks)
            @test any(c -> c.event == Symbol("response.completed"), cb.chunks)
        finally
            close(server)
        end
    end
end
