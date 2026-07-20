# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).


"""
    colgram_operator(A::MatrixShaped)

Returns a matrix-shaped operator `G == A' * A`, the column-Gram
construction of a matrix or matrix-shaped operator `A`.

Typically returns a [`ColGramOperator`](@ref):

```julia
G = colgram_operator(A)
G * x == A' * (A * x)
```

`G` is hermitian (symmetric for real element types) and positive
semi-definite by construction; `isposdef` is `true` only if `A` is
structurally known to have full column rank.

`colgram_factor(colgram_operator(A))` must return an equivalent of
[`asoperator`](@ref)`(A)`.

Equivalent to [`rowgram_operator`](@ref)`(A')`.
"""
colgram_operator(A::MatrixShaped) = ColGramOperator(A)
export colgram_operator


"""
    struct ColGramOperator{T<:Number,OP} <: MatrixShapedOperator{T}

Represents the column-Gram operator `A' * A` of a matrix or
matrix-shaped operator `A`, storing only [`asoperator`](@ref)`(A)`
internally.

User code should not call `ColGramOperator` directly, but use
[`colgram_operator`](@ref) instead.

Keeps structured factors (`UpperTriangular`, `Diagonal`, ...) as such,
their structure drives trait derivation and [`lower_cholesky`](@ref).
"""
struct ColGramOperator{T<:Number,OP} <: MatrixShapedOperator{T}
    A::OP
end
export ColGramOperator

function ColGramOperator(A::MatrixShaped)
    op = asoperator(A)
    return ColGramOperator{eltype(op),typeof(op)}(op)
end

"""
    colgram_factor(op)

Returns a factor `G` with `op == G' * G`, always as a matrix-shaped
operator.

See also [`rowgram_factor`](@ref) with the same contract (no triangular
structure, canonical form or uniqueness guaranteed).
"""
colgram_factor(op::MatrixShapedOperator) = adjoint(rowgram_factor(op))
colgram_factor(op::ColGramOperator) = op.A
export colgram_factor

rowgram_factor(op::ColGramOperator) = adjoint(op.A)

Base.size(op::ColGramOperator) = (size(op.A, 2), size(op.A, 2))

Base.adjoint(op::ColGramOperator) = op

Symmetricity(::ColGramOperator{T}) where T =
    NumberDomain(T) === Real ? IsSymmetric() : UnknownSymmetricity()
Hermitianity(::ColGramOperator) = IsHermitian()

# Derivations delegate through the adjoint identity
# colgram_operator(A) == rowgram_operator(A'):
_as_rowgram(op::ColGramOperator) = rowgram_operator(adjoint(op.A))

PosDefNess(op::ColGramOperator) = PosDefNess(_as_rowgram(op))

# The stored factor is always an operator and argument shapes were
# checked by the outer entry point, so component application stays in
# the unchecked implementation layer:
explicit_mul_impl(op::ColGramOperator, v::AbstractVector{<:Number}) =
    explicit_mul_impl(adjoint(op.A), explicit_mul_impl(op.A, v))
batched_mul_impl(op::ColGramOperator, X::AbstractMatrix{<:Number}) =
    explicit_mul_impl(adjoint(op.A), explicit_mul_impl(op.A, X))

LinearAlgebra.isposdef(op::ColGramOperator) = isposdef(_as_rowgram(op))

LinearAlgebra.logabsdet(op::ColGramOperator) = logabsdet(_as_rowgram(op))

Base.:(==)(a::ColGramOperator, b::ColGramOperator) = a.A == b.A

Base.hash(op::ColGramOperator, h::UInt) = hash(op.A, hash(:ColGramOperator, h))

Base.isapprox(a::ColGramOperator, b::ColGramOperator; kwargs...) = isapprox(a.A, b.A; kwargs...)

function Base.show(io::IO, op::ColGramOperator)
    print(io, "colgram_operator(")
    show(io, op.A)
    print(io, ")")
end


ldiv_impl(op::ColGramOperator, x::AbstractVecOrMat{<:Number}) = ldiv_impl(_as_rowgram(op), x)

lower_cholesky(op::ColGramOperator) = lower_cholesky(_as_rowgram(op))
