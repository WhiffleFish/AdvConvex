mat = get_spam_data()
X_train, Y_train, X_test, Y_test = train_test_split(mat, 0.334)

w0 = rand(size(X_train, 1))

f = LogRegProblem(X_train,Y_train)
∇f(x) = HW3.∇(f,x)

@btime f($w0)
@btime ∇f($w0)

prob = DifferentiableProblem(f, ∇f)
solver = GradientDescentSolver(
    α = 1e-6,
    ϵ = 1e-4,
    linesearch = BackTrackingLineSearch(),
)
w0 = zeros(size(X_test, 1))
w_opt, hist = solve(solver, prob, w0)
plot(hist.f)
plot(hist.f,yscale=:log10, xscale=:log10)

train_acc = map(hist.x) do x
    HW3.accuracy(x, X_train, Y_train)
end

test_acc = map(hist.x) do x
    HW3.accuracy(x, X_test, Y_test)
end

plot(train_acc, xscale=:log10, label="train", ylabel="Accuracy", xlabel="Optimization Step (log scale)")
plot!(test_acc, label="test")

HW3.accuracy(w_opt, X_test, Y_test)
HW3.accuracy(w_opt, X_train, Y_train)

perms = sortperm(abs.(w_opt))

bar(
    abs.(w_opt[perms]), label="",
    xticks=(eachindex(perms),perms),
    xrotation=45, xtickfont=(4), xlims=(0,Inf),
    xlabel = "feature index (i)",
    ylabel = L"|w_i|",
    minorgrid = false,
    title = "Feature Importance"
)
