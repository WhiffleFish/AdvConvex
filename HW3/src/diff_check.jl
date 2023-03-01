function basis(n, i)
    e = zeros(n)
    e[i] = 1
    return e
end

forward_diff(f, x::Number, h) = (f(x + h) - f(x)) / h

function forward_diff(f, x, h)
    f_x = f(x)
    n = length(x)
    ∇̂f = zero(x)
    for i ∈ eachindex(x)
        ∇̂f[i] = (f(x + basis(n,i)*h) - f_x) / h
    end
    return ∇̂f
end

function forward_diff_grad_error(f, ∇f, x, h)
    ∇̂f = forward_diff(f, x, h)
    return norm(∇f(x) .- ∇̂f, 2)
end

function forward_diff_grad_error(f, ∇f, x, h::AbstractVector)
    ϵ = zero(h)
    for i ∈ eachindex(h)
        ϵ[i] = forward_diff_grad_error(f, ∇f, x, h[i])
    end
    return ϵ
end

centered_diff(f, x::Number, h) = (f(x + h) - f(x - h)) / 2h

function centered_diff(f, x::AbstractVector, h)
    n = length(x)
    ∇̂f = zero(x)
    for i ∈ eachindex(x)
        ∇̂f[i] = (f(x + basis(n,i)*h) - f(x - basis(n,i)*h)) / 2h
    end
    return ∇̂f
end

function centered_diff_grad_error(f, ∇f, x, h::Number)
    ∇̂f = centered_diff(f, x, h)
    return norm(∇f(x) .- ∇̂f, 2)
end

function centered_diff_grad_error(f, ∇f, x, h::AbstractVector)
    ϵ = zero(h)
    for i ∈ eachindex(h)
        ϵ[i] = centered_diff_grad_error(f, ∇f, x, h[i])
    end
    return ϵ
end
