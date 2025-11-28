using StreamCallbacks
using Test
using Aqua
using HTTP, JSON3
using StreamCallbacks: build_response_body, is_done, extract_chunks, print_content,
                       callback, handle_error_message, extract_content, streamed_request!
using StreamCallbacks: AbstractStreamFlavor, OpenAIStream, OpenAIChatStream, OpenAIResponsesStream,
                       AnthropicStream, StreamChunk, StreamCallback, OllamaStream

@testset "StreamCallbacks.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(StreamCallbacks)
    end
    include("interface.jl")
    include("shared_methods.jl")
    include("stream_openai_chat.jl")
    include("stream_openai_responses.jl")
    include("stream_anthropic.jl")
    include("stream_ollama.jl")
    include("integration_mock_server.jl")
    include("promptingtools_compatibility.jl")
end
