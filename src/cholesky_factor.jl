# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).


"""
    lower_cholesky(A)

Returns the lower Cholesky factor `L` of `A`, so that `A == L * L'`.

Unlike a general [`rowgram_factor`](@ref), the Cholesky factor has
lower-triangular structure with non-negative diagonal - positive for
positive definite `A`, where the factor is unique.

Defined where the factor is available structurally or cheaply,
implicit operators are never materialized or factorized. Structurally
triangular Gram factors are canonicalized by column sign resp. phase
factors. Matrix arguments yield matrix factors, operator arguments
yield operator factors.

See [`upper_cholesky`](@ref) for the adjoint factor.
"""
function lower_cholesky end
export lower_cholesky

lower_cholesky(A::AbstractMatrix{<:Number}) = lower_cholesky(cholesky(A))

# The value check rejects non-factorizable diagonals for both number
# domains, matching the exception `cholesky` throws for matrices; the
# factor of a real-non-negative diagonal is real:
function lower_cholesky(A::Diagonal{<:Number})
    all(x -> isreal(x) && real(x) >= 0, A.diag) || throw(PosDefException(1))
    return Diagonal(sqrt.(real.(A.diag)))
end

lower_cholesky(C::Cholesky) = C.L

lower_cholesky(op::MatrixShapedOperator) = throw(ArgumentError(
    "no structural lower_cholesky available for operator of type $(nameof(typeof(op)))"
))

_canonical_chol(op::MatrixShapedOperator) = throw(ArgumentError(
    "no structural lower_cholesky available for operator of type $(nameof(typeof(op)))"
))

LinearAlgebra.logabsdet(op::MatrixShapedOperator) = throw(ArgumentError(
    "no structural logabsdet available for operator of type $(nameof(typeof(op)))"
))


"""
    upper_cholesky(A)

Returns the upper Cholesky factor `U` of `A`, so that `A == U' * U`.

The resulting Cholesky factor is the adjoint of [`lower_cholesky`](@ref).
"""
upper_cholesky(A) = adjoint(lower_cholesky(A))
export upper_cholesky
