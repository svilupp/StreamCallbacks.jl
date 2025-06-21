# LibCURL-based streaming implementation for StreamCallbacks.jl
using LibCURL

"""
    curl_write_callback(ptr::Ptr{UInt8}, size::Csize_t, nmemb::Csize_t, userdata::Ptr{Cvoid})

Callback function for processing streaming response data from libcurl.
"""
function curl_write_callback(ptr::Ptr{UInt8}, size::Csize_t, nmemb::Csize_t, userdata::Ptr{Cvoid})::Csize_t
    callback_data = unsafe_pointer_to_objref(userdata)
    cb, spillover_ref, isdone_ref, verbose = callback_data[]
    
    # Read the data
    data_size = size * nmemb
    chunk_data = unsafe_string(ptr, data_size)
    
    # Extract chunks using existing logic
    chunks, new_spillover = extract_chunks(
        cb.flavor, chunk_data; verbose, spillover=spillover_ref, cb.kwargs...)
    
    # Update spillover
    callback_data[] = (cb, new_spillover, isdone_ref, verbose)
    
    # Process chunks
    for chunk in chunks
        verbose && @debug "Chunk Data: $(chunk.data)"
        handle_error_message(chunk; throw_on_error=cb.throw_on_error, verbose, cb.kwargs...)
        if is_done(cb.flavor, chunk; verbose, cb.kwargs...)
            callback_data[] = (cb, new_spillover, true, verbose)
        end
        callback(cb, chunk)
        push!(cb, chunk)
    end
    
    return data_size
end

"""
    curl_header_callback_impl(ptr::Ptr{UInt8}, size::Csize_t, nmemb::Csize_t, userdata::Ptr{Cvoid})

Callback function for processing response headers from libcurl.
"""
function curl_header_callback_impl(ptr::Ptr{UInt8}, size::Csize_t, nmemb::Csize_t, userdata::Ptr{Cvoid})::Csize_t
    header_data = unsafe_pointer_to_objref(userdata)
    response_headers, status_code = header_data[]
    
    # Read header line
    header_size = size * nmemb
    header_line = unsafe_string(ptr, header_size)
    header_line = strip(header_line)
    
    # Parse status line
    if startswith(header_line, "HTTP/")
        parts = split(header_line)
        length(parts) >= 2 && (status_code[] = parse(Int, parts[2]))
    elseif occursin(":", header_line)
        # Parse header
        colon_pos = findfirst(':', header_line)
        if !isnothing(colon_pos)
            key = strip(header_line[1:colon_pos-1])
            value = strip(header_line[colon_pos+1:end])
            response_headers[lowercase(key)] = value
        end
    end
    
    return header_size
end

"""
    libcurl_streamed_request!(cb::AbstractStreamCallback, url::String, headers::Vector, body::String; kwargs...)

LibCURL-based implementation of streamed_request! with better performance and reliability.
"""
function libcurl_streamed_request!(cb::AbstractStreamCallback, url::String, headers::Vector, body::String; kwargs...)
    verbose = get(kwargs, :verbose, false) || cb.verbose
    
    # Initialize curl handle
    curl = LibCURL.curl_easy_init()
    curl == C_NULL && error("Failed to initialize curl")
    
    # Response data collection
    response_headers = Dict{String,String}()
    status_code = Ref{Int}(0)
    spillover = ""
    isdone = false
    header_list = C_NULL

    try
        # Set basic options
        LibCURL.curl_easy_setopt(curl, LibCURL.CURLOPT_URL, url)
        LibCURL.curl_easy_setopt(curl, LibCURL.CURLOPT_POST, 1)
        LibCURL.curl_easy_setopt(curl, LibCURL.CURLOPT_POSTFIELDS, body)
        LibCURL.curl_easy_setopt(curl, LibCURL.CURLOPT_POSTFIELDSIZE, length(body))
        
        # Set headers
        for (key, value) in headers
            header_str = "$key: $value"
            header_list = LibCURL.curl_slist_append(header_list, header_str)
        end
        header_list != C_NULL && LibCURL.curl_easy_setopt(curl, LibCURL.CURLOPT_HTTPHEADER, header_list)
        
        # Write callback for streaming response data
        write_callback = @cfunction(curl_write_callback, Csize_t, (Ptr{UInt8}, Csize_t, Csize_t, Ptr{Cvoid}))
        callback_data = Ref((cb, spillover, isdone, verbose))
        LibCURL.curl_easy_setopt(curl, LibCURL.CURLOPT_WRITEFUNCTION, write_callback)
        LibCURL.curl_easy_setopt(curl, LibCURL.CURLOPT_WRITEDATA, pointer_from_objref(callback_data))
        
        # Header callback for response headers
        header_callback = @cfunction(curl_header_callback_impl, Csize_t, (Ptr{UInt8}, Csize_t, Csize_t, Ptr{Cvoid}))
        header_data = Ref((response_headers, status_code))
        LibCURL.curl_easy_setopt(curl, LibCURL.CURLOPT_HEADERFUNCTION, header_callback)
        LibCURL.curl_easy_setopt(curl, LibCURL.CURLOPT_HEADERDATA, pointer_from_objref(header_data))
        
        # SSL options
        LibCURL.curl_easy_setopt(curl, LibCURL.CURLOPT_CAINFO, LibCURL.cacert)
        LibCURL.curl_easy_setopt(curl, LibCURL.CURLOPT_SSL_VERIFYPEER, 1)
        LibCURL.curl_easy_setopt(curl, LibCURL.CURLOPT_SSL_VERIFYHOST, 2)
        
        # Perform the request
        res = LibCURL.curl_easy_perform(curl)        

        res != LibCURL.CURLE_OK && error("curl_easy_perform failed: $(LibCURL.curl_easy_strerror(res))")
        
        # Get final status code
        status_ref = Ref{Clong}(0)
        LibCURL.curl_easy_getinfo(curl, LibCURL.CURLINFO_RESPONSE_CODE, status_ref)
        final_status = Int(status_ref[])
        
        # Verify content type
        content_type = get(response_headers, "content-type", "")
        if cb.flavor isa OllamaStream
            @assert occursin("application/x-ndjson", lowercase(content_type)) "For OllamaStream flavor, Content-Type must be application/x-ndjson"
        else
            @assert occursin("text/event-stream", lowercase(content_type)) "Content-Type must be text/event-stream"
        end
        
        # Aesthetic newline for stdout
        cb.out == stdout && (println(); flush(stdout))
        
        # Build response body
        body_content = build_response_body(cb.flavor, cb; verbose, cb.kwargs...)
        
        # Create response object
        resp = (
            status = final_status,
            headers = collect(response_headers),
            body = JSON3.write(body_content)
        )
        
        return resp
        
    finally
        # Cleanup
        header_list != C_NULL && LibCURL.curl_slist_free_all(header_list)
        LibCURL.curl_easy_cleanup(curl)
    end
end

"""
    libcurl_streamed_request!(cb::AbstractStreamCallback, url::String, headers::Vector, body::IOBuffer; kwargs...)

LibCURL-based implementation that accepts IOBuffer input.
"""
libcurl_streamed_request!(cb::AbstractStreamCallback, url::String, headers::Vector, body::IOBuffer; kwargs...) = 
    libcurl_streamed_request!(cb, url, headers, String(take!(body)); kwargs...)