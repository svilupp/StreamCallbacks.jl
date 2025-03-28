# Simulate some data
blob = "event: start\ndata: {\"key\": \"value\"}\n\nevent: end\ndata: {\"status\": \"complete\"}\n\n"
chunks, spillover = extract_chunks(OpenAIStream(), blob)
blob = "{\"key\":\"value\", \"done\":true}"
chunks, spillover = extract_chunks(OllamaStream(), blob)
# Add Gemini precompilation
gemini_blob = "{\"candidates\": [{\"content\": {\"parts\": [{\"text\": \"Hello\"}],\"role\": \"model\"}}],\"usageMetadata\": {\"promptTokenCount\": 9,\"totalTokenCount\": 9},\"modelVersion\": \"gemini-2.0-flash\"}\r\n\r\n"
chunks, spillover = extract_chunks(GoogleStream(), gemini_blob)

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

# GoogleStream examples
flavor = GoogleStream()
gemini_chunk = StreamChunk(
    nothing,
    """{"candidates":[{"content":{"parts":[{"text":"Hello"}],"role":"model"}}]}""",
    JSON3.read("""{"candidates":[{"content":{"parts":[{"text":"Hello"}],"role":"model"}}]}""")
)
is_done(flavor, gemini_chunk)
extract_content(flavor, gemini_chunk)
cb_gemini = StreamCallback(out = io, flavor = GoogleStream())
callback(cb_gemini, gemini_chunk)
build_response_body(flavor, cb_gemini)
