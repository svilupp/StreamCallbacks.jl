module StreamCallbacks

using HTTP, JSON3
using PrecompileTools

export StreamCallback, StreamChunk, OpenAIStream, OpenAIChatStream, OpenAIResponsesStream,
       AnthropicStream, OllamaStream, streamed_request!
include("interface.jl")

include("shared_methods.jl")

include("stream_openai_chat.jl")

include("stream_openai_responses.jl")

include("stream_anthropic.jl")

include("stream_ollama.jl")

@compile_workload begin
    include("precompilation.jl")
end

end
