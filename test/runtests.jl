using StreamCallbacks
using Test
using Aqua
using HTTP, JSON3
using StreamCallbacks: build_response_body, is_done, extract_chunks, print_content,
                       callback, handle_error_message, extract_content
using StreamCallbacks: AbstractStreamFlavor, OpenAIStream, AnthropicStream, StreamChunk,
                       StreamCallback

@testset "StreamCallbacks.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(StreamCallbacks)
    end
    include("interface.jl")
    include("shared_methods.jl")
    include("stream_openai.jl")
    include("stream_anthropic.jl")
end
