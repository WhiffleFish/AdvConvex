using LinearAlgebra
using AdvConvex.HW3
using Plots

h_vec = 10 .^ (range(-15,stop=1,length=50))
# 1d linear
f_lin(x) = 2x + 1
∇f_lin(x) = 2

# 1d quadratic
f_quad(x) = 3x^2 + 2x + 1
∇f_quad(x) = 6x + 2

# 1d cubic
f_cube(x) = 4x^3 + 3x^2 + 2x + 1
∇f_cube(x) = 12x^2 + 6x + 2


const A_test = rand(100,100)

f_mat_quad(x) = dot(x, A_test, x)
∇f_mat_quad(x) = 2A_test*x
∇f_mat_quad_wrong(x) = A_test*x

plot(h_vec, centered_diff_grad_error(f_lin, ∇f_lin, 0, h_vec), xscale=:log10, label="centered diff")
plot!(h_vec, forward_diff_grad_error(f_lin, ∇f_lin, 0, h_vec), xscale=:log10, label="forward diff")

plot(h_vec, centered_diff_grad_error(f_quad, ∇f_quad, 0, h_vec), xscale=:log10, label="centered diff")
plot!(h_vec, forward_diff_grad_error(f_quad, ∇f_quad, 0, h_vec), xscale=:log10, label="forward diff")

plot(h_vec, centered_diff_grad_error(f_cube, ∇f_cube, 3, h_vec), xscale=:log10, yscale=:log10, label="centered diff")
plot!(h_vec, forward_diff_grad_error(f_cube, ∇f_cube, 3, h_vec), xscale=:log10, yscale=:log10, label="forward diff")

x_test = rand(size(A_test, 1))
plot(h_vec, centered_diff_grad_error(f_mat_quad, ∇f_mat_quad, x_test, h_vec), xscale=:log10, yscale=:log10, label="centered diff")
plot!(h_vec, forward_diff_grad_error(f_mat_quad, ∇f_mat_quad, x_test, h_vec), xscale=:log10, yscale=:log10, label="forward diff")

plot(h_vec, centered_diff_grad_error(f_mat_quad, ∇f_mat_quad_wrong, x_test, h_vec), xscale=:log10, yscale=:log10, label="centered diff")
plot!(h_vec, forward_diff_grad_error(f_mat_quad, ∇f_mat_quad_wrong, x_test, h_vec), xscale=:log10, yscale=:log10, label="forward diff")



mat = get_spam_data()
X_train, Y_train, X_test, Y_test = train_test_split(mat, 0.05)
@assert size(X_train, 2) + size(X_test, 2) == size(mat, 2)

w = rand(size(X_train, 1))

logistic_loss_grad(w, X_train, Y_train)

l(w) = logistic_loss(w, X_train, Y_train)
∇l(w) = logistic_loss_grad(w, X_train, Y_train)


using BenchmarkTools
@btime logistic_loss_grad($w, $X_train, $Y_train)



plot(h_vec, centered_diff_grad_error(l, ∇l, w, h_vec), xscale=:log10, yscale=:log10, label="centered diff")
plot!(h_vec, forward_diff_grad_error(l, ∇l, w, h_vec), xscale=:log10, yscale=:log10, label="forward diff")


##
X = X_train
Y = Y_train
logistic_loss_grad(w, X, Y)

n,m = size(X)
cache = HW3.LogRegCache(n,m)
using BenchmarkTools

lc(w) = HW3.logistic_loss!(cache, w, X_train, Y_train)
∇lc(w) = HW3.logistic_loss_grad!(cache, w, X_train, Y_train)

lc(w)
l(w)

w = randn(size(X,1))*10

∇lc(w)
plot(
    h_vec,
    centered_diff_grad_error(lc, ∇lc, w, h_vec),
    xscale=:log10, yscale=:log10, label="centered diff"
)
plot!(
    h_vec,
    forward_diff_grad_error(lc, ∇lc, w, h_vec),
    label="forward diff"
)
@btime HW3.logistic_loss_grad!($cache, $w, $X, $Y)

@benchmark HW3.logistic_loss_grad!($cache, $w, $X, $Y)


##

f = LogRegProblem(X,Y)
prob = DifferentiableProblem(f, x->HW3.∇(f,x))
solver = GradientDescentSolver(α=1e-4, max_iter=10_000, linesearch=BackTrackingLineSearch())
w0 = rand(size(X,1))
w_opt, hist = solve(solver, prob, w0)

plot(hist.f,yscale=:log10)

train_preds = map(σ.(X'*w_opt)) do x
    x < 0.5 ? -1 : 1
end

sum(train_preds .== Y_train) / length(Y_train)

test_preds = map(σ.(X_test'*w_opt)) do x
    x < 0.5 ? -1 : 1
end

sum(test_preds .== Y_test) / length(Y_test)


acc_hist = zeros(length(hist.x))
for i ∈ eachindex(hist.x)
    w = hist.x[i]
    preds = map(σ.(X_test'*w)) do x
        x < 0.5 ? -1 : 1
    end
    acc_hist[i] = sum(preds .== Y_test) / length(Y_test)
end

plot(acc_hist)

plot(norm.(hist.∇f,2), yscale=:log10)

perms = sortperm(abs.(w_opt))
bar(
    abs.(w_opt[perms]), label="",
    xticks=(eachindex(perms),perms),
    xrotation=45, xtickfont=(4), xlims=(0,Inf),
    xlabel = "feature index (i)",
    ylabel = L"|w_i|",
    minorgrid = false
)
