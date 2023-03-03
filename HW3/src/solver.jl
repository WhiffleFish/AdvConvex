Base.@kwdef struct GradientDescentSolver{LS}
    α::Float64      = 1e-3
    ϵ::Float64      = 1e-5
    max_iter::Int   = typemax(Int)
    linesearch::LS  = NoLineSearch()
end

struct NoLineSearch end

(::NoLineSearch)(f, ∇f, x, t) = x - t*∇f(x)

Base.@kwdef struct BackTrackingLineSearch
    ρ::Float64  = 0.90
    c::Float64  = 1e-4
end

function (ls::BackTrackingLineSearch)(f, ∇f, x, t)
    (;ρ,c) = ls
    ∇f_x =∇f(x)
    p_k = -∇f_x
    f_x = f(x)
    while f(x + t*p_k) > f_x + c*t*dot(∇f_x, p_k)
        t *= ρ
    end
    return x - t*∇f_x
end

Base.@kwdef struct SolverHistory
    x::Vector{Vector{Float64}}  = Vector{Float64}[]
    f::Vector{Float64}          = Float64[]
    ∇f::Vector{Vector{Float64}} = Vector{Float64}[]
end

function Base.push!(hist::SolverHistory, x, f, ∇f)
    push!(hist.x, x)
    push!(hist.f, f)
    push!(hist.∇f, ∇f)
end

struct DifferentiableProblem{F,GF}
    f::F
    ∇f::GF
end

function solve(sol::GradientDescentSolver, p::DifferentiableProblem, x0)
    (;f, ∇f) = p
    (;α, ϵ, linesearch) = sol
    hist = SolverHistory()
    x = copy(x0)
    f_x = f(x)
    ∇f_x = ∇f(x)
    push!(hist, copy(x), copy(f_x), copy(∇f_x))

    for i ∈ 1:sol.max_iter
        x = linesearch(f, ∇f, x, α)
        ∇f_x .= ∇f(x)
        f_x_new = f(x)
        push!(hist, copy(x), copy(f_x), copy(∇f_x))
        Δf = f_x_new - f_x
        f_x = f_x_new
        Δf > 0. && @warn "objective increase"
        abs(Δf) < ϵ && break
    end
    return x, hist
end
