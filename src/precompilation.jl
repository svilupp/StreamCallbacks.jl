# Simulate some data
blob = "event: start\ndata: {\"key\": \"value\"}\n\nevent: end\ndata: {\"status\": \"complete\"}\n\n"
chunks, spillover = extract_chunks(OpenAIStream(), blob)
blob = "{\"key\":\"value\", \"done\":true}"
chunks, spillover = extract_chunks(OllamaStream(), blob)

# Chunk examples
io = IOBuffer()
cb = StreamCallback(out = io, flavor = OpenAIStream())
example_chunk = StreamChunk(
    nothing,
    """{"choices":[{"delta":{"content":"Hello"}}]}""",
    JSON3.read("""{"choices":[{"delta":{"content":"Hello"}}]}""")
)

# OpenAIStream examples
flavor = OpenAIStream()
is_done(flavor, example_chunk)
extract_content(flavor, example_chunk)
callback(cb, example_chunk)
build_response_body(flavor, cb)

# AnthropicStream examples
flavor = AnthropicStream()
is_done(flavor, example_chunk)
extract_content(flavor, example_chunk)
build_response_body(flavor, cb)

# OllamaStream examples
flavor = OllamaStream()
is_done(flavor, example_chunk)
extract_content(flavor, example_chunk)
build_response_body(flavor, cb)
