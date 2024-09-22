using StreamCallbacks
using Test
using Aqua
using HTTP, JSON3

@testset "StreamCallbacks.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(StreamCallbacks)
    end
    # Write your tests here.
    include("shared_methods.jl")
    include("stream_openai.jl")
    include("stream_anthropic.jl")
end
