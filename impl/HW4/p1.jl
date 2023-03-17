using Plots
using AdvConvex.HW3
using AdvConvex.HW4
using Optim

mat = get_spam_data()
X_train, Y_train, X_test, Y_test = train_test_split(mat, 0.05)
w = rand(size(X_train, 1))

f = LogRegProblem(X_test,Y_test)
∇f(w) = HW3.∇(f, w)

prob = DifferentiableProblem(f, ∇f)
nest_solver = NesterovDescentSolver(
    α = 1e-2,
    ϵ = 0.0,
    max_iter = 10^4,
    linesearch = BackTrackingLineSearch(),
)
w0 = zeros(size(X_test, 1))
w_opt_nest, hist_nest = HW4.solve(nest_solver, prob, w0)

gd_solver = GradientDescentSolver(
    α = 1e-3,
    ϵ = 1e-10,
    max_iter = 10^4,
    linesearch = BackTrackingLineSearch(),
)
w0 = zeros(size(X_test, 1))
w_opt_gd, hist_gd = HW3.solve(gd_solver, prob, w0)

w0 = zeros(size(X_test, 1))
nm_solver = NelderMead()
res = optimize(f, w0, nm_solver,
    Optim.Options(iterations=10_000, show_trace=false, store_trace=true)
)



plot(
    hist_nest.f, yscale=:log10,
    label="nesterov descent", lw=2,
    xlabel="Iteration", ylabel="f(x)",
    ylims=(10^(floor(log10(last(hist_nest.f)))),Inf), yminorgrid=true)
plot!(hist_gd.f, label="gradient descent", lw=2)
plot!(getfield.(res.trace, :value), label="nelder-mead", lw=2)
savefig(joinpath(@__DIR__, "p1.pdf"))
