module StreamCallbacks

using HTTP, JSON3
using PrecompileTools

export StreamCallback, StreamChunk, streamed_request!
export OpenAIStream, AnthropicStream, GoogleStream, OllamaStream

include("interface.jl")

include("shared_methods.jl")

include("stream_openai.jl")

include("stream_anthropic.jl")

include("stream_ollama.jl")

include("stream_gemini.jl")

@compile_workload begin
    include("precompilation.jl")
end

end
