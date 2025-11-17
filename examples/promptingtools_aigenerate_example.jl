using PromptingTools
using StreamCallbacks
using GoogleGenAI
using PromptingTools: GoogleSchema

# Stream to stdout with callback collecting chunks
cb = StreamCallback(out = stdout, verbose = false)

# Use a real model id; adjust as needed
msg = @time aigenerate(GoogleSchema(), "Tell me a short story of humanity:";
                 model = "gemini-2.5-pro-preview-06-05",
                 streamcallback = cb)

println("\n\nFinal content:\n", msg.content)
#%%
using GoogleGenAI

models = list_models()
for m in models
    if "createCachedContent" in m[:supported_generation_methods]
        println(m[:name])
    end
end

