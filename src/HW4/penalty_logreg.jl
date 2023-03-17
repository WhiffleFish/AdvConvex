struct PenaltyLogRegProblem
    logreg::LogRegProblem
    λ::Float64
end

g(l::PenaltyLogRegProblem,w) = l.logreg(w)
∇g(l,w) = HW3.∇(l.logreg, w)
h(l::PenaltyLogRegProblem,w) = l.λ * norm(w, 1)
prox_th(l::PenaltyLogRegProblem, t, y) = sign(y)*max(abs(y) - t*l.λ, 0.0)

f(l::PenaltyLogRegProblem, w) = g(l,w) + h(l,w)
