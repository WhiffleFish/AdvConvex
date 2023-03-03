σ(x) = inv(1 + exp(-x))

μ(w::AbstractVector,x::AbstractVector, y::Number) = σ(y*dot(w, x))
μ(w::AbstractVector, X::AbstractMatrix, y::AbstractVector) = σ.( y .* (X' * w))


struct LogRegCache
    _mul1::Vector{Float64}
    _mul2::Vector{Float64}
    LogRegCache(n,m) = new(Vector{Float64}(undef, m), Vector{Float64}(undef, n))
end

function logistic_loss(w::AbstractVector, X::Matrix, y::AbstractVector)
    @assert size(X,2) == length(y)
    l = 0.0
    for (x_i, y_i) ∈ zip(eachcol(X), y)
        l += log(1 + exp(-y_i*dot(w, x_i)))
    end
    return l
end

function logistic_loss!(cache::LogRegCache, w::AbstractVector, X::Matrix, y::AbstractVector)
    @assert size(X,2) == length(y)
    tmp1 = mul!(cache._mul1, X', w)
    @. tmp1 = log(1 + exp(-y * tmp1))
    return sum(tmp1)
end

# function logistic_loss_grad(w::AbstractVector, X::Matrix, y::AbstractVector)
#     n = length(w)
#     @assert n == size(X,1)
#     @assert size(X,2) == length(y)
#     ∇l = zero(w)
#     for (x_i, y_i) ∈ zip(eachcol(X), y)
#         ∇l += σ(-y_i*dot(w, x_i))*y_i*x_i
#     end
#     return -∇l
# end

logistic_loss_grad(w::AbstractVector, X::Matrix, y::AbstractVector) = -X*(y .* (1 .- μ(w, X, y)))

function logistic_loss_grad!(cache::LogRegCache, w::AbstractVector, X::Matrix, y::AbstractVector)
    tmp1 = mul!(cache._mul1, X', w) # n_samples
    @. tmp1 = y * (1 - σ( y * tmp1)) # n_samples
    return mul!(cache._mul2, X, tmp1) .*= -1 # x_dim
end


struct LogRegProblem
    X::Matrix{Float64}
    Y::Vector{Float64}
    cache::LogRegCache
    LogRegProblem(X,Y) = new(X,Y, LogRegCache(size(X)...))
end

(f::LogRegProblem)(w) = logistic_loss!(f.cache, w, f.X, f.Y)
∇(f::LogRegProblem, w) = logistic_loss_grad!(f.cache, w, f.X, f.Y)
