# Use
#
#     DOCUMENTER_DEBUG=true julia --color=yes make.jl local [nonstrict] [fixdoctests]
#
# for local builds.

using Documenter
using MatrixShapedOperators

# Doctest setup
DocMeta.setdocmeta!(
    MatrixShapedOperators,
    :DocTestSetup,
    :(using MatrixShapedOperators);
    recursive = true
)

makedocs(
    sitename = "MatrixShapedOperators",
    modules = [MatrixShapedOperators],
    format = Documenter.HTML(
        prettyurls = !("local" in ARGS),
        canonical = "https://oschulz.github.io/MatrixShapedOperators.jl/stable/"
    ),
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
        "LICENSE" => "LICENSE.md"
    ],
    doctest = ("fixdoctests" in ARGS) ? :fix : true,
    linkcheck = !("nonstrict" in ARGS),
    warnonly = ("nonstrict" in ARGS)
)

deploydocs(
    repo = "github.com/oschulz/MatrixShapedOperators.jl.git",
    forcepush = true,
    push_preview = true
)
