#!/usr/bin/env julia

using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(PackageSpec(path=joinpath(@__DIR__, "..")))
Pkg.instantiate()

using CairoMakie
using Distributions
using MCMCChains
using MetropolisHastingsMCMC
using Random
using Colors
CairoMakie.activate!(; type="png")

rng = Random.MersenneTwister(20240707)
μ, σ = 0.0, 0.6
target = LogNormal(μ, σ)

function log_p_lognormal(x)
    x1 = first(x)
    x1 > zero(x1) ? logpdf(target, x1) : -Inf
end

result = mh_mcmc(log_p_lognormal,
                 [1.0];
                 n_iter=6_000,
                 σ=0.45,
                 target_acceptance_rate=0.30,
                 κ=0.65,
                 n_iter_adaptation=3_000,
                 rng=rng)

samples = result.samples[1_001:end, :]
chain = Chains(samples, [:x])

trace_color = RGBf(0.211, 0.113, 0.533)
density_color = RGBf(0.376, 0.588, 0.898)

fig = Figure(; size=(1200, 650), fontsize=20)

ax_trace = Axis(fig[1, 1];
                title="Metropolis-Hastings trace",
                xlabel="Iteration",
                ylabel="x")
lines!(ax_trace, 1:size(samples, 1), samples[:, 1]; color=trace_color, linewidth=2)
hideydecorations!(ax_trace; grid=false)

ax_density = Axis(fig[1, 2];
                  title="Posterior density",
                  xlabel="x",
                  ylabel="Density")
density!(ax_density, samples[:, 1]; color=density_color, strokewidth=3, label="MH samples")

xgrid = range(0.0, stop=exp(μ + 3σ), length=300)
lines!(ax_density, xgrid, pdf.(target, xgrid);
       color=:black, linewidth=2, linestyle=:dash,
       label="LogNormal(μ=$μ, σ=$σ)")
axislegend(ax_density, position=:rt)

Label(fig[0, 1:2],
      "MetropolisHastingsMCMC.jl × CairoMakie.jl",
      fontsize=26,
      color=trace_color,
      halign=:center,
      font=:bold)

Label(fig[2, 1:2],
      # "Inspired by Turing.jl visualization patterns",
      fontsize=18,
      color=RGBA(0.2, 0.2, 0.2, 0.8),
      halign=:center)

output_path = joinpath(@__DIR__, "lognormal_mh.png")
save(output_path, fig; px_per_unit=2.0)
@info "Saved figure" output_path
