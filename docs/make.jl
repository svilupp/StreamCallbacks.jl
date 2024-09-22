using StreamCallbacks
using Documenter

DocMeta.setdocmeta!(
    StreamCallbacks, :DocTestSetup, :(using StreamCallbacks); recursive = true)

makedocs(;
    modules = [StreamCallbacks],
    authors = "J S <49557684+svilupp@users.noreply.github.com> and contributors",
    sitename = "StreamCallbacks.jl",
    format = Documenter.HTML(;
        canonical = "https://svilupp.github.io/StreamCallbacks.jl",
        edit_link = "main",
        assets = String[]
    ),
    pages = [
        "Home" => "index.md",
        "API" => "api.md"
    ]
)

deploydocs(;
    repo = "github.com/svilupp/StreamCallbacks.jl",
    devbranch = "main"
)
