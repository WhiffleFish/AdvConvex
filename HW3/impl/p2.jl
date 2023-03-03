using LinearAlgebra
using HW3
using Plots
using Test

struct LLSProblem
    A::Matrix{Float64}
    b::Vector{Float64}
    cache::Vector{Float64}
    LLSProblem(A,b) = new(A,b,similar(b))
end

(f::LLSProblem)(x) = 0.5*norm(f.A*x - f.b,2)^2
function ∇(f::LLSProblem, x)
    (;A,b) = f
    return A' * (A*x - b)
end

h_vec = 10 .^ (range(-15,stop=10,length=50))

n = 10
m = 100
A = rand(m, n)
b = rand(m)

f = LLSProblem(A,b)
∇f(x) = ∇(f,x)

x = randn(n)*10
A*x - b
f(x)
∇f(x)


plot(h_vec, centered_diff_grad_error(f, ∇f, x, h_vec), xscale=:log10, yscale=:log10, label="centered diff")
plot!(h_vec, forward_diff_grad_error(f, ∇f, x, h_vec), xscale=:log10, yscale=:log10, label="forward diff")

prob = DifferentiableProblem(f, ∇f)
sol = GradientDescentSolver(α=inv(opnorm(A)^2),ϵ=1e-15)

x0 = randn(n)
x_opt,hist = solve(sol, prob, x0)
true_x_opt = pinv(A)*b

p = plot(hist.f, yscale=:log10)
Plots.abline!(p, 0.0, f(true_x_opt))

@test true_x_opt ≈ x_opt atol=1e-6

## backtracking
sol = GradientDescentSolver(
    α = inv(opnorm(A)^2),
    ϵ = 1e-15,
    linesearch = BackTrackingLineSearch()
)
x_opt,hist = solve(sol, prob, x0)

p = plot(hist.f, xscale=:log10, yscale=:log10)
Plots.abline!(p, 0.0, f(true_x_opt))

@test true_x_opt ≈ x_opt atol=1e-6
