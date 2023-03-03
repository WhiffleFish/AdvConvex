using LinearAlgebra
using HW3
using Plots
using BenchmarkTools

h_vec = 10 .^ (range(-15,stop=1,length=100))
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

#TODO: "Is your code robust to scaling f and g to be large or small?"
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
w = rand(size(X_train, 1))

f = LogRegProblem(X_test,Y_test)
∇f(w) = HW3.∇(f, w)

plot(
    h_vec,
    centered_diff_grad_error(f, ∇f, w, h_vec),
    xscale=:log10, yscale=:log10, label="centered diff",
    xlabel = L"h", ylabel = "error"
)
plot!(
    h_vec,
    forward_diff_grad_error(f, ∇f, w, h_vec),
    label="forward diff"
)
