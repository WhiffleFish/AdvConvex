Base.@kwdef struct NesterovDescentSolver{LS}
    α::Float64          = 1e-3
    ρ::Float64          = 0.90
    v::Vector{Float64}  = Float64[]
    ϵ::Float64          = 1e-5
    max_iter::Int       = typemax(Int)
    linesearch::LS      = NoLineSearch()
end

function solve(sol::NesterovDescentSolver, p::DifferentiableProblem, x0)
    (;f, ∇f) = p
    (;α, ρ, ϵ, v, linesearch) = sol
    hist = SolverHistory()
    resize!(v, length(x0))
    v .= 0.0
    x = copy(x0)
    f_x = f(x)
    ∇f_x = ∇f(x)
    _step = ∇f_x
    push!(hist, copy(x), copy(f_x), copy(∇f_x))

    for i ∈ 1:sol.max_iter
        d = @. ρ^2 * v - (1-ρ) * α * ∇f_x
        @. v = ρ*v - α*∇f_x
        @. _step = -d
        x = linesearch(f, _step, x, 1.0)
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
