# Shared test utilities and fixture loaders
#
# IMPORTANT: These helpers use the actual extract_chunks function from the library
# to ensure we're testing real parsing behavior, not a reimplementation.

# Helper to load fixture and parse into chunks using the real parser
function load_responses_fixture(filename)
    filepath = joinpath(@__DIR__, "fixtures", filename)
    content = read(filepath, String)
    chunks, _ = extract_chunks(OpenAIResponsesStream(), content)
    return chunks
end

# Helper to load fixture for Chat Completions API using the real parser
function load_chat_fixture(filename)
    filepath = joinpath(@__DIR__, "fixtures", filename)
    content = read(filepath, String)
    chunks, _ = extract_chunks(OpenAIChatStream(), content)
    return chunks
end
