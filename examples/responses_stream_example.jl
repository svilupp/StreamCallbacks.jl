using HTTP
using JSON3
using OpenAI
using PromptingTools
const PT = PromptingTools
using PromptingTools: OpenAIResponseSchema, AbstractResponseSchema, airespond
using StreamCallbacks: StreamCallback

# This example demonstrates the use of the OpenAI Responses API
# with proper schema support and streaming capabilities

# Make sure your OpenAI API key is set in the environment variable OPENAI_API_KEY

# Basic usage with the new schema
schema = OpenAIResponseSchema()
cb = StreamCallback(out=stdout)

response = airespond(schema, "What is the 6th largest city in the Czech Republic? you can think, but in the answer I only want to see the city."; 
model = "gpt-5.1-codex", streamcallback=cb)
@show response.tokens
@show response.extras[:usage]