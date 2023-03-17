const λ = 5.0
mat = get_spam_data()
X_train, Y_train, X_test, Y_test = train_test_split(mat, 0.334)

f = LogRegProblem(X_train,Y_train)
∇f(x) = HW3.∇(f,x)
prob = DifferentiableProblem(f, ∇f)
solver = GradientDescentSolver(
    α = 1e-6,
    ϵ = 1e-4,
    max_iter = 1000,
    linesearch = BackTrackingLineSearch(),
)
w_opt1, hist1 = solve(solver, prob, w0)

l = PenaltyLogRegProblem(f, λ)

p = ProximalProblem(
    w -> HW4.f(l, w),
    w -> HW4.∇g(l, w),
    (y,t=5e-4) -> HW4.prox_th(l,t,y)
)

w0 = ones(size(X_test, 1))
solver = GradientDescentSolver(
    α = 1e-5,
    ϵ = 0.0,
    max_iter=1000
)

w_opt2, hist2 = HW4.solve(solver, p, w0)

plot(hist1.f, yscale=:log10)
plot(hist2.f)

w_opt2 - w_opt1

plot(w_opt2)
plot!(w_opt1)
test_acc1 = map(hist1.x) do x
    HW3.accuracy(x, X_test, Y_test)
end

test_acc2 = map(hist2.x) do x
    HW3.accuracy(x, X_test, Y_test)
end
plot(test_acc1)
plot!(test_acc2)

plot(test_acc2)
