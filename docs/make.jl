using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(PackageSpec(path=joinpath(@__DIR__, "..")))
Pkg.instantiate()

using MetropolisHastingsMCMC
using Documenter

DocMeta.setdocmeta!(MetropolisHastingsMCMC, :DocTestSetup, :(using MetropolisHastingsMCMC); recursive=true)

makedocs(;
    modules=[MetropolisHastingsMCMC],
    authors="Andreas Scheidegger <andreas.scheidegger@eawag.ch> and contributors",
    repo="https://github.com/scheidan/MetropolisHastingsMCMC.jl/blob/{commit}{path}#{line}",
    sitename="MetropolisHastingsMCMC.jl",
    checkdocs=:exports,
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://scheidan.github.io/MetropolisHastingsMCMC.jl",
        assets=String[],
    ),
    pages=[
        "Metropolis-Hastings MCMC" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/scheidan/MetropolisHastingsMCMC.jl",
    devbranch="main",
)
