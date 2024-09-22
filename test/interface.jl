
@testset "StreamCallback" begin
    # Test default constructor
    cb = StreamCallback()
    @test cb.out == stdout
    @test isnothing(cb.flavor)
    @test isempty(cb.chunks)
    @test cb.verbose == false
    @test cb.throw_on_error == false
    @test isempty(cb.kwargs)

    # Test custom constructor
    custom_out = IOBuffer()
    custom_flavor = OpenAIStream()
    custom_chunks = [StreamChunk(event = :test, data = "test data")]
    custom_cb = StreamCallback(;
        out = custom_out,
        flavor = custom_flavor,
        chunks = custom_chunks,
        verbose = true,
        throw_on_error = true,
        kwargs = (custom_key = "custom_value",)
    )
    @test custom_cb.out == custom_out
    @test custom_cb.flavor == custom_flavor
    @test custom_cb.chunks == custom_chunks
    @test custom_cb.verbose == true
    @test custom_cb.throw_on_error == true
    @test custom_cb.kwargs == (custom_key = "custom_value",)

    # Test Base methods
    cb = StreamCallback()
    @test isempty(cb)
    push!(cb, StreamChunk(event = :test, data = "test data"))
    @test length(cb) == 1
    @test !isempty(cb)
    empty!(cb)
    @test isempty(cb)

    # Test show method
    cb = StreamCallback(out = IOBuffer(), flavor = OpenAIStream())
    str = sprint(show, cb)
    @test occursin("StreamCallback(out=IOBuffer", str)
    @test occursin("flavor=OpenAIStream()", str)
    @test occursin("silent, no_throw", str)

    chunk = StreamChunk(event = :test, data = "{\"a\": 1}", json = JSON3.read("{\"a\": 1}"))
    str = sprint(show, chunk)
    @test occursin("StreamChunk(event=test", str)
    @test occursin("data={\"a\": 1}", str)
    @test occursin("json keys=a", str)

    push!(cb, chunk)
    @test length(cb) == 1
    @test !isempty(cb)
    empty!(cb)
    @test isempty(cb)

    # Test configure_callback! method
    cb, api_kwargs = configure_callback!(StreamCallback(), OpenAISchema())
    @test cb.flavor isa OpenAIStream
    @test api_kwargs[:stream] == true
    @test api_kwargs[:stream_options] == (include_usage = true,)

    cb, api_kwargs = configure_callback!(StreamCallback(), AnthropicSchema())
    @test cb.flavor isa AnthropicStream
    @test api_kwargs[:stream] == true

    # Test error for unsupported schema
    @test_throws ErrorException configure_callback!(StreamCallback(), GoogleSchema())

    # Test configure_callback! with output stream
    cb, _ = configure_callback!(IOBuffer(), OpenAISchema())
    @test cb isa StreamCallback
    @test cb.out isa IOBuffer
    @test cb.flavor isa OpenAIStream
end