module MetropolisHastingsMCMC

export mh_mcmc

using ProgressMeter: Progress, next!, finish!
using TestItems
using Random: AbstractRNG, default_rng, rand, randn, randn!
using Statistics: mean, var
import LogDensityProblems

include("logdensityproblem_interface.jl")

"""
    mh_mcmc(lp, inits; n_iter, σ, target_acceptance_rate, κ, n_iter_adaptation, rng)

Adaptive random-walk Metropolis–Hastings sampler. Works with targets implementing
the `LogDensityProblems.jl` interface or plain log-density functions.

# Keyword Arguments
- `n_iter::Integer = 100`: number of MCMC iterations.
- `σ::Real = 2.4 / (length(inits)^(1 / 6))`: initial global scale of the random-walk proposal.
- `target_acceptance_rate::Real = 0.234`: desired acceptance rate guiding adaptation.
- `κ::Real = 0.6`: exponent of the Robbins–Monro learning rate, must be in `(0.5, 1]`.
- `n_iter_adaptation`: number of iterations during which adaptation is active (default `Inf`).
- `rng::AbstractRNG = default_rng()`: RNG used for proposals and acceptance decisions.

# Returns
Named tuple with fields
- `samples`: matrix of size `(n_iter, dimension)` containing the Markov chain.
- `log_p`: vector with the log-density evaluations.
"""
function mh_mcmc(lp,
                 inits::AbstractVector{T};
                 n_iter::Integer=100,
                 σ::Real=2.4 / (length(inits)^(1 / 6)),
                 target_acceptance_rate::Real=0.234,
                 κ::Real=0.6,
                 n_iter_adaptation=Inf,
                 rng::AbstractRNG=default_rng()) where {T<:Real}

    d = LogDensityProblems.dimension(lp)
    length(inits) == d ||
        error("The initial values must be of length $(d)!")

    state_type = float(T)
    initial_state = state_type.(inits)
    log_π₀ = LogDensityProblems.logdensity(lp, initial_state)
    log_type = float(promote_type(typeof(log_π₀), state_type))

    chain = Array{state_type}(undef, n_iter, d)
    chain[1, :] .= initial_state

    log_ps = Vector{log_type}(undef, n_iter)
    log_ps[1] = log_type(log_π₀)
    log_π = log_ps[1]

    proposal = Vector{state_type}(undef, d)
    noise = similar(proposal)

    log_σ = log(float(σ))

    progress = n_iter > 1 ? Progress(n_iter - 1; dt=1.0, desc="Sampling... ") : nothing

    for t in 2:n_iter
        @views x = chain[t - 1, :]

        σ_current = exp(log_σ)
        rw_proposal!(rng, proposal, noise, x, σ_current)

        log_πᵖ = log_type(LogDensityProblems.logdensity(lp, proposal))

        prob_accept = acceptance_prob(log_π, log_πᵖ)
        draw = rand(rng, float(typeof(prob_accept)))

        if draw < prob_accept
            chain[t, :] .= proposal
            log_π = log_ps[t] = log_πᵖ
        else
            chain[t, :] .= x
            log_ps[t] = log_π
        end

        if t <= n_iter_adaptation
            γ = t^(-κ)  # Robbins–Monro learning rate
            log_σ += γ * (Float64(prob_accept) - target_acceptance_rate)
        end

        if progress !== nothing
            next!(progress)
        end
    end

    if progress !== nothing
        finish!(progress)
    end

    return (samples=chain, log_p=log_ps)
end

# Wrapper when only a log-density function is given
function mh_mcmc(log_p::Function,
                 inits::AbstractVector{T};
                 kwargs...) where {T<:Real}
    lp = SimpleLogDensityProblem(log_p, length(inits))
    mh_mcmc(lp, inits; kwargs...)
end

"""
    rw_proposal!(rng, proposal, noise, x, σ)

In-place random-walk proposal. Fills `noise` with `𝒩(0, 1)` draws and writes the proposal
into `proposal` by adding `σ * noise` to `x`.
"""
function rw_proposal!(rng::AbstractRNG,
                      proposal::AbstractVector{T},
                      noise::AbstractVector{T},
                      x::AbstractVector{T},
                      σ::Real) where {T<:Real}
    randn!(rng, noise)
    σ_T = convert(T, σ)
    @inbounds @. proposal = x + σ_T * noise
    return proposal
end

"""
    acceptance_prob(log_π, log_πᵖ)

Metropolis–Hastings acceptance probability. Returns `1` when the proposal has
higher (log) density than the current state and `exp(Δ)` otherwise, where
`Δ = log_πᵖ - log_π`.
"""
function acceptance_prob(log_π::Real, log_πᵖ::Real)
    Δ = log_πᵖ - log_π
    return Δ ≥ zero(Δ) ? oneunit(Δ) : exp(Δ)
end

@testitem "Gaussian target moments" tags=[:distribution, :stochastic] begin
    using Random
    using Statistics
    rng = Random.MersenneTwister(42)
    log_p(x) = -0.5 * sum(abs2, x)
    inits = fill(3.0, 2)
    res = mh_mcmc(log_p, inits; n_iter=40_000, σ=1.0, κ=0.6, rng=rng)
    burnin = 5_000
    samples = res.samples[burnin+1:end, :]
    μ = vec(mean(samples, dims=1))
    σ² = vec(var(samples, dims=1))
    @test all(abs.(μ) .< 0.12)
    @test all(abs.(σ² .- 1.0) .< 0.15)
end

@testitem "MCMCTesting calibration" tags=[:calibration, :stochastic] begin
    using Random, Distributions, MCMCTesting
    using Statistics
    using ConcreteStructs: @concrete
    rng = Random.MersenneTwister(123)

    struct StdNormalModel end
    @concrete struct RWKernel{T<:Real}
        σ::T
    end

    logpdf_standard(x) = logpdf(Normal(0, 1), x)

    function MCMCTesting.sample_joint(rng::AbstractRNG, ::StdNormalModel)
        θ = randn(rng)
        y = randn(rng)
        return θ, y
    end

    function MCMCTesting.markovchain_transition(rng::AbstractRNG,
                                                ::StdNormalModel,
                                                kernel::RWKernel,
                                                θ::Float64,
                                                _y)
        proposal = θ + kernel.σ * randn(rng)
        log_current = logpdf_standard(θ)
        log_proposal = logpdf_standard(proposal)
        if rand(rng) < MetropolisHastingsMCMC.acceptance_prob(log_current, log_proposal)
            return proposal
        else
            return θ
        end
    end

    test = TwoSampleTest(150, 60)
    subject = TestSubject(StdNormalModel(), RWKernel(1.0))
    Random.seed!(rng, 42)
    @test seqmcmctest(test, subject, 0.05, 25; show_progress=false)
end

end
