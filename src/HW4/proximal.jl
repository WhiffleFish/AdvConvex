struct ProximalProblem{F,G,PH}
    f::F
    ∇g::G
    prox_h::PH
end

function solve(sol::GradientDescentSolver, p::ProximalProblem, x0)
    (;f, ∇g, prox_h) = p
    (;α, ϵ) = sol
    hist = SolverHistory()
    x = copy(x0)
    f_x = f(x)
    ∇g_x = ∇g(x)
    push!(hist, copy(x), copy(f_x), copy(∇g_x))

    for i ∈ 1:sol.max_iter
        G_t_x = (x - prox_h.(x .- α*∇g_x)) / α
        x = x - α*G_t_x
        f_x_new = f(x)
        ∇g_x = ∇g(x)
        push!(hist, copy(x), copy(f_x), copy(G_t_x))
        Δf = f_x_new - f_x
        f_x = f_x_new
        Δf > 0. && @warn "objective increase"
        abs(Δf) < ϵ && break
    end
    return x, hist
end
