# LibCURL-based streaming implementation for StreamCallbacks.jl
using LibCURL

"""
    stream_write_callback(ptr::Ptr{UInt8}, size::Csize_t, nmemb::Csize_t, userdata::Ptr{Cvoid})

Callback function for processing streaming response data from libcurl.
"""
function stream_write_callback(ptr::Ptr{UInt8}, size::Csize_t, nmemb::Csize_t, userdata::Ptr{Cvoid})::Csize_t
    callback_data = unsafe_pointer_to_objref(userdata)
    cb, spillover_ref, isdone_ref, verbose, error_body = callback_data[]
    
    # Read the data
    data_size = size * nmemb
    chunk_data = unsafe_string(ptr, data_size)
    
    # Always capture raw data for potential error responses
    write(error_body, chunk_data)
    
    # Extract chunks using existing logic
    chunks, new_spillover = extract_chunks(
        cb.flavor, chunk_data; verbose, spillover=spillover_ref, cb.kwargs...)
    
    # Update spillover
    callback_data[] = (cb, new_spillover, isdone_ref, verbose, error_body)
    
    # Process chunks
    for chunk in chunks
        verbose && @debug "Chunk Data: $(chunk.data)"
        handle_error_message(chunk; throw_on_error=cb.throw_on_error, verbose, cb.kwargs...)
        if is_done(cb.flavor, chunk; verbose, cb.kwargs...)
            callback_data[] = (cb, new_spillover, true, verbose, error_body)
        end
        callback(cb, chunk)
        push!(cb, chunk)
    end
    
    return data_size
end

"""
    stream_header_callback(ptr::Ptr{UInt8}, size::Csize_t, nmemb::Csize_t, userdata::Ptr{Cvoid})

Callback function for processing response headers from libcurl.
"""
function stream_header_callback(ptr::Ptr{UInt8}, size::Csize_t, nmemb::Csize_t, userdata::Ptr{Cvoid})::Csize_t
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
    streamed_request_libcurl!(cb::AbstractStreamCallback, url::String, headers::Vector, body::String; kwargs...)

LibCURL-based implementation of streamed_request! with better performance and reliability.
"""
function streamed_request_libcurl!(cb::AbstractStreamCallback, url::String, headers::Vector, body::String; kwargs...)
    verbose = get(kwargs, :verbose, false) || cb.verbose
    
    # Initialize curl handle
    curl = LibCURL.curl_easy_init()
    curl == C_NULL && error("Failed to initialize curl")
    
    # Response data collection
    response_headers = Dict{String,String}()
    status_code = Ref{Int}(0)
    spillover = ""
    isdone = false
    error_body = IOBuffer()
    header_list = C_NULL

    try
        # Set basic options
        LibCURL.curl_easy_setopt(curl, LibCURL.CURLOPT_URL, url)
        LibCURL.curl_easy_setopt(curl, LibCURL.CURLOPT_POST, 1)
        LibCURL.curl_easy_setopt(curl, LibCURL.CURLOPT_POSTFIELDS, body)
        LibCURL.curl_easy_setopt(curl, LibCURL.CURLOPT_POSTFIELDSIZE, sizeof(body))
        
        # Set headers
        for (key, value) in headers
            header_str = "$key: $value"
            header_list = LibCURL.curl_slist_append(header_list, header_str)
        end
        header_list != C_NULL && LibCURL.curl_easy_setopt(curl, LibCURL.CURLOPT_HTTPHEADER, header_list)
        
        # Write callback for streaming response data
        write_callback = @cfunction(stream_write_callback, Csize_t, (Ptr{UInt8}, Csize_t, Csize_t, Ptr{Cvoid}))
        callback_data = Ref((cb, spillover, isdone, verbose, error_body))
        LibCURL.curl_easy_setopt(curl, LibCURL.CURLOPT_WRITEFUNCTION, write_callback)
        LibCURL.curl_easy_setopt(curl, LibCURL.CURLOPT_WRITEDATA, pointer_from_objref(callback_data))
        
        # Header callback for response headers
        header_callback = @cfunction(stream_header_callback, Csize_t, (Ptr{UInt8}, Csize_t, Csize_t, Ptr{Cvoid}))
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
        
        # Check for HTTP error status codes first
        if final_status >= 400
            content_type = get(response_headers, "content-type", "")
            error_body_str = String(take!(error_body))
            
            error_msg = """
            HTTP Error $(final_status): Request failed
            Response headers:\n - $(join(["$k: $v" for (k,v) in response_headers], "\n - "))"""
            
            if occursin("application/json", lowercase(content_type)) && !isempty(error_body_str)
                error_msg *= "\nError response body: $(error_body_str)"
            end
            
            error(error_msg * "\nPlease check your request parameters, API key, and model availability.")
        end
        
        # Verify content type for successful responses
        content_type = get(response_headers, "content-type", "")
        expected_type = cb.flavor isa OllamaStream ? "application/x-ndjson" : "text/event-stream"
        
        if !occursin(expected_type, lowercase(content_type))
            flavor_name = cb.flavor isa OllamaStream ? "OllamaStream" : "streaming"
            error("""
            For $(flavor_name) flavor, Content-Type must be $(expected_type).
            Received type: $(content_type)
            Status code: $(final_status)
            Response headers:\n - $(join(["$k: $v" for (k,v) in response_headers], "\n - "))
            Please check the model you are using and that you set `stream=true`.
            """)
        end
        
        # Aesthetic newline for stdout
        cb.out == stdout && (println(); flush(stdout))
        
        # Build response body
        body_content = build_response_body(cb.flavor, cb; verbose, cb.kwargs...)
        
        # Create response object
        resp = (
            status = Int16(final_status),
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
    streamed_request_libcurl!(cb::AbstractStreamCallback, url::String, headers::Vector, body::IOBuffer; kwargs...)

LibCURL-based implementation that accepts IOBuffer input.
"""
streamed_request_libcurl!(cb::AbstractStreamCallback, url::String, headers::Vector, body::IOBuffer; kwargs...) = 
    streamed_request_libcurl!(cb, url, headers, String(take!(body)); kwargs...)