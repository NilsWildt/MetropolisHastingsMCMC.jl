
import LogDensityProblems
using ConcreteStructs: @concrete
"""
    SimpleLogDensityProblem(log_p, dim)

Wrapper that exposes a plain log-density function through the `LogDensityProblems.jl` API.
"""
@concrete struct SimpleLogDensityProblem{F}
    log_p::F
    dim::Int
end

LogDensityProblems.dimension(lp::SimpleLogDensityProblem) = lp.dim

function LogDensityProblems.capabilities(::Type{SimpleLogDensityProblem})
    LogDensityProblems.LogDensityOrder{0}()
end

LogDensityProblems.logdensity(lp::SimpleLogDensityProblem, x) = lp.log_p(x)
