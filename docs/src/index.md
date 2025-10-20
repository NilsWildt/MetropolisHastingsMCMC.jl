```@meta
CurrentModule = MetropolisHastingsMCMC
```

# MetropolisHastingsMCMC.jl

`MetropolisHastingsMCMC.jl` provides a lightweight adaptive random-walk Metropolis–Hastings sampler.
It keeps the convenient `LogDensityProblems.jl` interface used by the Barker sampler while dropping the
gradient requirement.

## Quick start

### `LogDensityProblems.jl` targets

```julia
using MetropolisHastingsMCMC
using TransformVariables: transform, inverse, as, asℝ, asℝ₊
using TransformedLogDensities: TransformedLogDensity
using LogDensityProblemsAD: ADgradient
using Distributions

x = rand(10)
y = 2 .+ 1.5x .+ randn(10) * 0.2

model(x, θ) = θ.a + θ.b * x

function likelihood(θ, x, y)
    sum(logpdf(Normal(model(x[i], θ), θ.σ), y[i]) for i in eachindex(y))
end

prior(θ) = logpdf(Normal(0, 1), θ.a) +
           logpdf(Normal(0, 1), θ.b) +
           logpdf(Exponential(1), θ.σ)

posterior(θ, x, y) = likelihood(θ, x, y) + prior(θ)

trans = as((a = asℝ, b = asℝ, σ = asℝ₊))
lp = ADgradient(:ForwardDiff, TransformedLogDensity(trans, θ -> posterior(θ, x, y)))

inits = inverse(trans, (a = 0.0, b = 0.0, σ = 0.5))
results = mh_mcmc(lp, inits; n_iter = 5_000)

transform.(Ref(trans), eachrow(results.samples))  # samples in the original parameterisation
```

### Direct log-density functions

```julia
using MetropolisHastingsMCMC

banana(x) = -0.05 * (100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2)

res = mh_mcmc(banana, [5.0, -5.0]; n_iter = 2_000)
res.samples
res.log_p
```

## API reference

```@docs
mh_mcmc
```
