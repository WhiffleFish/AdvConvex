module HW3
using LinearAlgebra
using Downloads
using DelimitedFiles
using Random

include("diff_check.jl")
export forward_diff_grad_error, centered_diff_grad_error

include("logistic.jl")
export σ, μ, logistic_loss, logistic_loss_grad, logistic_loss_hess

include("spam_data.jl")
export get_spam_data, train_test_split

end
