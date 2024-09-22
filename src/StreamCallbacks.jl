module StreamCallbacks

using HTTP, JSON3

export StreamCallback, StreamChunk, OpenAIStream, AnthropicStream, streamed_request!
include("interface.jl")

include("shared_methods.jl")

include("stream_openai.jl")

include("stream_anthropic.jl")

end
