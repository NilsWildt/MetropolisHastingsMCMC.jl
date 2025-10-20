using MetropolisHastingsMCMC
using TestItems
using TestItemRunner
using ConcreteStructs: @concrete

const _PKGROOT = normpath(joinpath(@__DIR__, ".."))

@testitem "Random walk adapts scale" tags=[:adaptation, :stochastic] begin
    using MetropolisHastingsMCMC, Random, Statistics

    rng = Random.MersenneTwister(2024)
    log_p(x) = -0.5 * sum(abs2, x)
    inits = fill(5.0, 3)

    result = mh_mcmc(log_p, inits;
                     n_iter=8_000,
                     σ=5.0,
                     target_acceptance_rate=0.3,
                     κ=0.7,
                     n_iter_adaptation=5_000,
                     rng=rng)

    burn = 1_000
    samples = result.samples[burn+1:end, :]
    moves = count(i -> any(result.samples[i, :] .!= result.samples[i - 1, :]),
                  2:size(result.samples, 1))
    acc_rate = moves / (size(result.samples, 1) - 1)
    @test acc_rate > 0.15 && acc_rate < 0.45

    μ = vec(mean(samples, dims=1))
    @test all(abs.(μ) .< 0.25)
end

@testitem "LogDensityProblems workflow" tags=[:integration, :stochastic] begin
    using MetropolisHastingsMCMC
    using TransformVariables: transform, inverse, as, asℝ, asℝ₊
    using TransformedLogDensities: TransformedLogDensity
    using LogDensityProblemsAD: ADgradient
    using Distributions, Random, Statistics

    rng = Random.MersenneTwister(7)
    x = rand(rng, 20)
    y = 1.5 .+ 0.7x .+ randn(rng, 20) .* 0.15

    model(x, θ) = θ.a .+ θ.b .* x
    function loglik(θ, x, y)
        sum(logpdf(Normal(model(x[i], θ), θ.σ), y[i]) for i in eachindex(y))
    end
    prior(θ) = logpdf(Normal(0, 2), θ.a) +
               logpdf(Normal(0, 2), θ.b) +
               logpdf(Exponential(1), θ.σ)
    posterior(θ, x, y) = loglik(θ, x, y) + prior(θ)

    trans = as((a=asℝ, b=asℝ, σ=asℝ₊))
    lp = ADgradient(:ForwardDiff, TransformedLogDensity(trans, θ -> posterior(θ, x, y)))
    inits = inverse(trans, (a=0.0, b=0.0, σ=0.5))

    result = mh_mcmc(lp, inits; n_iter=6_000, rng=rng)
    samples = result.samples[5_001:end, :]
    transformed = transform.(Ref(trans), eachrow(samples))

    a_vals = map(s -> s.a, transformed)
    b_vals = map(s -> s.b, transformed)
    σ_vals = map(s -> s.σ, transformed)

    @test mean(a_vals) ≈ 1.5 atol=0.25
    @test mean(b_vals) ≈ 0.7 atol=0.25
    @test mean(σ_vals) ≈ 0.15 atol=0.15
end

@testitem "Float32 state support" begin
    using MetropolisHastingsMCMC, Random, Statistics

    rng = Random.MersenneTwister(11)
    log_p(x) = -sum(abs2, x)
    inits = fill(Float32(0.5), 3)

    result = mh_mcmc(log_p, inits;
                     n_iter=5_000,
                     σ=Float32(0.9),
                     target_acceptance_rate=0.3,
                     rng=rng)

    @test eltype(result.samples) === Float32
    @test result.samples[1, 1] isa Float32
end

@testitem "BigFloat state support" begin
    using MetropolisHastingsMCMC

    setprecision(BigFloat, 256) do
        log_p(x) = -BigFloat(0.5) * sum(abs2, x)
        inits = fill(BigFloat(0.1), 2)
        result = mh_mcmc(log_p, inits; n_iter=500, σ=BigFloat(0.75))
        @test eltype(result.samples) === BigFloat
        @test eltype(result.log_p) === BigFloat
    end
end

const _ALLOWED_PREFIXES = map(dir -> normpath(joinpath(_PKGROOT, dir)),
                              ("src", "test"))

@run_package_tests filter=ti -> begin
    filename = get(ti, :filename, nothing)
    filename === nothing && return false
    path = normpath(String(filename))
    any(p -> startswith(path, p), _ALLOWED_PREFIXES)
end
