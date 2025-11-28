# PromptingTools.jl Compatibility Tests
#
# These tests ensure the response structures from build_response_body are compatible
# with PromptingTools.jl's response parsing functions.
#
# Fixture loading helpers are defined in test_utils.jl

# =============================================================================
# OpenAI Responses API (OpenAIResponsesStream)
# =============================================================================
# See: PromptingTools.jl/src/llm_openai_responses.jl extract_response_content()

@testset "PromptingTools-OpenAIResponsesStream" begin
    @testset "simple-response" begin
        flavor = OpenAIResponsesStream()
        chunks = load_responses_fixture("responses_api_simple.txt")

        cb = StreamCallback()
        cb.chunks = chunks
        response = build_response_body(flavor, cb)

        # Required top-level fields for PromptingTools
        @test haskey(response, :id)
        @test !isempty(response[:id])
        @test startswith(string(response[:id]), "resp_")

        @test haskey(response, :status)
        @test response[:status] == "completed"

        @test haskey(response, :model)
        @test !isempty(response[:model])

        # Reasoning field (even if empty for non-reasoning models)
        @test haskey(response, :reasoning)

        # Usage structure for token counting
        @test haskey(response, :usage)
        usage = response[:usage]
        @test haskey(usage, :input_tokens)
        @test haskey(usage, :output_tokens)
        @test usage[:input_tokens] > 0
        @test usage[:output_tokens] > 0

        # Output structure for extract_response_content()
        @test haskey(response, :output)
        @test response[:output] isa AbstractVector
        @test length(response[:output]) >= 1

        # Find message output item
        message_items = filter(item -> get(item, :type, "") == "message", response[:output])
        @test length(message_items) >= 1

        message_item = first(message_items)
        @test message_item[:type] == "message"
        @test haskey(message_item, :role)
        @test message_item[:role] == "assistant"
        @test haskey(message_item, :content)
        @test message_item[:content] isa AbstractVector

        # Content structure: [{type: "output_text", text: "..."}]
        content_items = message_item[:content]
        @test length(content_items) >= 1

        text_item = first(content_items)
        @test haskey(text_item, :type)
        @test text_item[:type] == "output_text"
        @test haskey(text_item, :text)
        @test !isempty(text_item[:text])
    end

    @testset "reasoning-response" begin
        flavor = OpenAIResponsesStream()
        chunks = load_responses_fixture("responses_api_reasoning.txt")

        cb = StreamCallback()
        cb.chunks = chunks
        response = build_response_body(flavor, cb)

        # Required top-level fields
        @test haskey(response, :id)
        @test haskey(response, :status)
        @test response[:status] == "completed"

        # Reasoning field should have effort/summary for reasoning models
        @test haskey(response, :reasoning)
        reasoning = response[:reasoning]
        @test haskey(reasoning, :effort)
        @test haskey(reasoning, :summary)

        # Usage with reasoning tokens
        @test haskey(response, :usage)
        usage = response[:usage]
        @test haskey(usage, :input_tokens)
        @test haskey(usage, :output_tokens)
        @test haskey(usage, :output_tokens_details)
        @test haskey(usage[:output_tokens_details], :reasoning_tokens)
        @test usage[:output_tokens_details][:reasoning_tokens] > 0

        # Output structure should have both reasoning and message items
        @test haskey(response, :output)
        @test length(response[:output]) >= 2

        # Find reasoning output item
        reasoning_items = filter(item -> get(item, :type, "") == "reasoning", response[:output])
        @test length(reasoning_items) >= 1

        reasoning_item = first(reasoning_items)
        @test reasoning_item[:type] == "reasoning"

        # Reasoning summary structure: [{type: "summary_text", text: "..."}]
        @test haskey(reasoning_item, :summary)
        @test reasoning_item[:summary] isa AbstractVector
        @test length(reasoning_item[:summary]) >= 1

        summary_item = first(reasoning_item[:summary])
        @test haskey(summary_item, :type)
        @test summary_item[:type] == "summary_text"
        @test haskey(summary_item, :text)
        @test !isempty(summary_item[:text])

        # Find message output item
        message_items = filter(item -> get(item, :type, "") == "message", response[:output])
        @test length(message_items) >= 1

        message_item = first(message_items)
        @test message_item[:type] == "message"
        @test message_item[:role] == "assistant"
        @test haskey(message_item, :content)

        # Content structure
        content_items = message_item[:content]
        @test length(content_items) >= 1

        text_item = first(content_items)
        @test text_item[:type] == "output_text"
        @test haskey(text_item, :text)
        @test !isempty(text_item[:text])
    end

    @testset "required-fields" begin
        # Verify all fields required by PromptingTools.extract_response_content() exist
        flavor = OpenAIResponsesStream()

        # Simple response - message output only
        simple_chunks = load_responses_fixture("responses_api_simple.txt")
        cb_simple = StreamCallback()
        cb_simple.chunks = simple_chunks
        simple_response = build_response_body(flavor, cb_simple)

        # output[] items must have :type field
        for item in simple_response[:output]
            @test haskey(item, :type)
        end
        # message items must have :content array with :type and :text
        msg = first(filter(i -> i[:type] == "message", simple_response[:output]))
        @test haskey(msg, :content)
        for c in msg[:content]
            @test haskey(c, :type)
            @test haskey(c, :text)
        end

        # Reasoning response - both reasoning and message outputs
        reasoning_chunks = load_responses_fixture("responses_api_reasoning.txt")
        cb_reasoning = StreamCallback()
        cb_reasoning.chunks = reasoning_chunks
        reasoning_response = build_response_body(flavor, cb_reasoning)

        # reasoning items must have :summary array with :text
        reasoning_item = first(filter(i -> i[:type] == "reasoning", reasoning_response[:output]))
        @test haskey(reasoning_item, :summary)
        for s in reasoning_item[:summary]
            @test haskey(s, :text)
        end
    end
end

# =============================================================================
# OpenAI Chat Completions API (OpenAIChatStream)
# =============================================================================
# See: PromptingTools.jl/src/llm_openai.jl

@testset "PromptingTools-OpenAIChatStream" begin
    @testset "fixture-response" begin
        flavor = OpenAIChatStream()
        chunks = load_chat_fixture("chat_completions_simple.txt")

        cb = StreamCallback()
        cb.chunks = chunks
        response = build_response_body(flavor, cb)

        # Required top-level fields
        @test haskey(response, :id)
        @test !isempty(response[:id])
        @test startswith(string(response[:id]), "chatcmpl-")

        @test haskey(response, :object)
        @test response[:object] == "chat.completion"

        @test haskey(response, :model)
        @test !isempty(response[:model])

        @test haskey(response, :created)

        # Choices structure
        @test haskey(response, :choices)
        @test response[:choices] isa AbstractVector
        @test length(response[:choices]) >= 1

        choice = first(response[:choices])
        @test haskey(choice, :index)
        @test choice[:index] == 0

        @test haskey(choice, :message)
        message = choice[:message]
        @test haskey(message, :role)
        @test message[:role] == "assistant"
        @test haskey(message, :content)
        @test !isempty(message[:content])

        @test haskey(choice, :finish_reason)
        @test choice[:finish_reason] == "stop"

        # Usage structure
        @test haskey(response, :usage)
        usage = response[:usage]
        @test haskey(usage, :prompt_tokens)
        @test haskey(usage, :completion_tokens)
        @test haskey(usage, :total_tokens)
        @test usage[:prompt_tokens] > 0
        @test usage[:completion_tokens] > 0
        @test usage[:total_tokens] == usage[:prompt_tokens] + usage[:completion_tokens]
    end

    @testset "manual-response" begin
        # Test with manually constructed chunks
        flavor = OpenAIChatStream()

        cb = StreamCallback()
        push!(cb.chunks,
            StreamChunk(
                nothing,
                """{"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}""",
                JSON3.read("""{"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}""")
            ))
        push!(cb.chunks,
            StreamChunk(
                nothing,
                """{"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":" world!"},"finish_reason":null}]}""",
                JSON3.read("""{"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":" world!"},"finish_reason":null}]}""")
            ))
        push!(cb.chunks,
            StreamChunk(
                nothing,
                """{"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}""",
                JSON3.read("""{"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}""")
            ))

        response = build_response_body(flavor, cb)

        # Verify content assembly
        @test response[:choices][1][:message][:content] == "Hello world!"

        # Verify all required fields
        @test response[:id] == "chatcmpl-test"
        @test response[:object] == "chat.completion"
        @test response[:model] == "gpt-4o"
        @test response[:choices][1][:finish_reason] == "stop"
        @test response[:usage][:prompt_tokens] == 5
        @test response[:usage][:completion_tokens] == 2
    end
end

# =============================================================================
# Anthropic API (AnthropicStream)
# =============================================================================
# See: PromptingTools.jl/src/llm_anthropic.jl

@testset "PromptingTools-AnthropicStream" begin
    @testset "simple-response" begin
        flavor = AnthropicStream()

        cb = StreamCallback(flavor = flavor)
        push!(cb.chunks,
            StreamChunk(
                :message_start,
                """{"type":"message_start","message":{"id":"msg_test123","type":"message","role":"assistant","content":[],"model":"claude-3-opus-20240229","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}""",
                JSON3.read("""{"type":"message_start","message":{"id":"msg_test123","type":"message","role":"assistant","content":[],"model":"claude-3-opus-20240229","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}""")
            ))
        push!(cb.chunks,
            StreamChunk(
                :content_block_start,
                """{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}""",
                JSON3.read("""{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}""")
            ))
        push!(cb.chunks,
            StreamChunk(
                :content_block_delta,
                """{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello, world!"}}""",
                JSON3.read("""{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello, world!"}}""")
            ))
        push!(cb.chunks,
            StreamChunk(
                :content_block_stop,
                """{"type":"content_block_stop","index":0}""",
                JSON3.read("""{"type":"content_block_stop","index":0}""")
            ))
        push!(cb.chunks,
            StreamChunk(
                :message_delta,
                """{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}""",
                JSON3.read("""{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}""")
            ))
        push!(cb.chunks,
            StreamChunk(
                :message_stop,
                """{"type":"message_stop"}""",
                JSON3.read("""{"type":"message_stop"}""")
            ))

        response = build_response_body(flavor, cb)

        # Required top-level fields for PromptingTools
        @test haskey(response, :model)
        @test response[:model] == "claude-3-opus-20240229"

        @test haskey(response, :stop_reason)
        @test response[:stop_reason] == "end_turn"

        @test haskey(response, :stop_sequence)

        # Content structure: [{type: "text", text: "..."}]
        @test haskey(response, :content)
        @test response[:content] isa AbstractVector
        @test length(response[:content]) >= 1

        content_item = first(response[:content])
        @test haskey(content_item, :type)
        @test content_item[:type] == "text"
        @test haskey(content_item, :text)
        @test content_item[:text] == "Hello, world!"

        # Usage structure
        @test haskey(response, :usage)
        usage = response[:usage]
        @test haskey(usage, :input_tokens)
        @test haskey(usage, :output_tokens)
        @test usage[:input_tokens] == 10
        @test usage[:output_tokens] == 5
    end

    @testset "thinking-response" begin
        # Test response with thinking content (extended thinking)
        flavor = AnthropicStream()

        cb = StreamCallback(flavor = flavor)
        push!(cb.chunks,
            StreamChunk(
                :message_start,
                """{"type":"message_start","message":{"id":"msg_think","type":"message","role":"assistant","content":[],"model":"claude-3-7-sonnet-20250219","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":15,"output_tokens":0}}}""",
                JSON3.read("""{"type":"message_start","message":{"id":"msg_think","type":"message","role":"assistant","content":[],"model":"claude-3-7-sonnet-20250219","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":15,"output_tokens":0}}}""")
            ))
        # Thinking block
        push!(cb.chunks,
            StreamChunk(
                :content_block_start,
                """{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}""",
                JSON3.read("""{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}""")
            ))
        push!(cb.chunks,
            StreamChunk(
                :content_block_delta,
                """{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me think about this..."}}""",
                JSON3.read("""{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me think about this..."}}""")
            ))
        push!(cb.chunks,
            StreamChunk(
                :content_block_stop,
                """{"type":"content_block_stop","index":0}""",
                JSON3.read("""{"type":"content_block_stop","index":0}""")
            ))
        # Text block
        push!(cb.chunks,
            StreamChunk(
                :content_block_start,
                """{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}""",
                JSON3.read("""{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}""")
            ))
        push!(cb.chunks,
            StreamChunk(
                :content_block_delta,
                """{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"The answer is 42."}}""",
                JSON3.read("""{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"The answer is 42."}}""")
            ))
        push!(cb.chunks,
            StreamChunk(
                :content_block_stop,
                """{"type":"content_block_stop","index":1}""",
                JSON3.read("""{"type":"content_block_stop","index":1}""")
            ))
        push!(cb.chunks,
            StreamChunk(
                :message_delta,
                """{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":20}}""",
                JSON3.read("""{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":20}}""")
            ))

        response = build_response_body(flavor, cb)

        # Should have both thinking and text content blocks
        @test haskey(response, :content)
        @test length(response[:content]) >= 1

        # Find text content (PromptingTools primarily uses text content)
        text_items = filter(c -> get(c, :type, "") == "text", response[:content])
        @test length(text_items) >= 1
        @test first(text_items)[:text] == "The answer is 42."
    end
end

# =============================================================================
# Ollama API (OllamaStream)
# =============================================================================
# See: PromptingTools.jl/src/llm_ollama.jl

@testset "PromptingTools-OllamaStream" begin
    @testset "simple-response" begin
        flavor = OllamaStream()

        cb = StreamCallback(flavor = flavor)
        push!(cb.chunks,
            StreamChunk(
                nothing,
                """{"model":"llama2","created_at":"2024-01-01T00:00:00Z","message":{"role":"assistant","content":"Hello"},"done":false}""",
                JSON3.read("""{"model":"llama2","created_at":"2024-01-01T00:00:00Z","message":{"role":"assistant","content":"Hello"},"done":false}""")
            ))
        push!(cb.chunks,
            StreamChunk(
                nothing,
                """{"model":"llama2","created_at":"2024-01-01T00:00:01Z","message":{"role":"assistant","content":" from"},"done":false}""",
                JSON3.read("""{"model":"llama2","created_at":"2024-01-01T00:00:01Z","message":{"role":"assistant","content":" from"},"done":false}""")
            ))
        push!(cb.chunks,
            StreamChunk(
                nothing,
                """{"model":"llama2","created_at":"2024-01-01T00:00:02Z","message":{"role":"assistant","content":" Ollama!"},"done":true,"prompt_eval_count":8,"eval_count":4,"total_duration":1000000000}""",
                JSON3.read("""{"model":"llama2","created_at":"2024-01-01T00:00:02Z","message":{"role":"assistant","content":" Ollama!"},"done":true,"prompt_eval_count":8,"eval_count":4,"total_duration":1000000000}""")
            ))

        response = build_response_body(flavor, cb)

        # Required top-level fields for PromptingTools
        @test haskey(response, :model)
        @test response[:model] == "llama2"

        @test haskey(response, :created_at)

        # Message structure
        @test haskey(response, :message)
        message = response[:message]
        @test haskey(message, :role)
        @test message[:role] == "assistant"
        @test haskey(message, :content)
        @test message[:content] == "Hello from Ollama!"

        # Done flag (note: reflects first chunk's value, not final state)
        @test haskey(response, :done)

        # Usage/token counts
        @test haskey(response, :prompt_eval_count)
        @test response[:prompt_eval_count] == 8
        @test haskey(response, :eval_count)
        @test response[:eval_count] == 4
    end

    @testset "with-context" begin
        # Test response with context (for conversation continuity)
        flavor = OllamaStream()

        cb = StreamCallback(flavor = flavor)
        push!(cb.chunks,
            StreamChunk(
                nothing,
                """{"model":"mistral","created_at":"2024-01-01T00:00:00Z","message":{"role":"assistant","content":"Test response"},"done":true,"context":[1,2,3,4,5],"prompt_eval_count":5,"eval_count":2}""",
                JSON3.read("""{"model":"mistral","created_at":"2024-01-01T00:00:00Z","message":{"role":"assistant","content":"Test response"},"done":true,"context":[1,2,3,4,5],"prompt_eval_count":5,"eval_count":2}""")
            ))

        response = build_response_body(flavor, cb)

        # Context should be preserved for conversation continuity
        @test haskey(response, :context)
        @test response[:context] == [1, 2, 3, 4, 5]

        @test response[:message][:content] == "Test response"
    end
end
