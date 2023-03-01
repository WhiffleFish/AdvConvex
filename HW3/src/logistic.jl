σ(x) = inv(1 + exp(-x))

μ(w::AbstractVector,x::AbstractVector, y::Number) = σ(y*dot(w, x))
μ(w::AbstractVector, X::AbstractMatrix, y::AbstractVector) = σ.( y .* (X' * w))

function logistic_loss(w::AbstractVector, X::Matrix, y::AbstractVector)
    @assert size(X,2) == length(y)
    l = 0.0
    for (x_i, y_i) ∈ zip(eachcol(X), y)
        l += log(1 + exp(-y_i*dot(w, x_i)))
    end
    return l
end

function logistic_loss_grad(w::AbstractVector, X::Matrix, y::AbstractVector)
    n = length(w)
    @assert n == size(X,1)
    @assert size(X,2) == length(y)
    ∇l = zero(w)
    for (x_i, y_i) ∈ zip(eachcol(X), y)
        ∇l += σ(-y_i*dot(w, x_i))*y_i*x_i
    end
    return -∇l
end

function logistic_loss_hess(w::AbstractVector, X::Matrix, y::AbstractVector)
    n = length(w)
    @assert n == size(X,1)
    @assert size(X,2) == length(y)
    ∇²l = zeros(n, n)
    for (x_i, y_i) ∈ zip(eachcol(X), y)
        μ_i = μ(w, x_i, y_i)
        ∇²l += μ_i*(1-μ_i) * x_i * x_i'
    end
    return ∇²l
end
