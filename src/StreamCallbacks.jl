module StreamCallbacks

using HTTP, JSON3
using PrecompileTools
using LibCURL

export StreamCallback, StreamChunk, OpenAIStream, AnthropicStream, OllamaStream,
       streamed_request!, libcurl_streamed_request!

include("interface.jl")

include("shared_methods.jl")
include("shared_methods_libcurl.jl")

include("stream_openai.jl")

include("stream_anthropic.jl")

include("stream_ollama.jl")

@compile_workload begin
    include("precompilation.jl")
end

end
