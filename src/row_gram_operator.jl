# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).


"""
    rowgram_operator(A::MatrixShaped)

Returns a matrix-shaped operator `G == A * A'`, the row-Gram
construction of a matrix or matrix-shaped operator `A`.

Typically returns a [`RowGramOperator`](@ref):

```julia
G = rowgram_operator(A)
G * x == A * (A' * x)
```

`G` is hermitian (symmetric for real element types) and positive
semi-definite by construction, and positive definite if `A` is
structurally known to have full row rank; `isposdef` also establishes
full rank cheaply for square triangular explicit factors.

`rowgram_factor(rowgram_operator(A))` must return an equivalent of
[`asoperator`](@ref)`(A)`.

Equivalent to [`colgram_operator`](@ref)`(A')`. Use this instead of the
plain product `A * A'`, which does not carry the Gram structure.
"""
rowgram_operator(A::MatrixShaped) = RowGramOperator(A)
export rowgram_operator


"""
    struct RowGramOperator{T<:Number,OP} <: MatrixShapedOperator{T}

Represents the row-Gram operator `A * A'` of a matrix or matrix-shaped
operator `A`, storing only [`asoperator`](@ref)`(A)` internally.

User code should not call `RowGramOperator` directly, but use
[`rowgram_operator`](@ref) instead.

Keeps structured factors (`LowerTriangular`, `Diagonal`, ...) as such,
their structure drives trait derivation and [`lower_cholesky`](@ref).
"""
struct RowGramOperator{T<:Number,OP} <: MatrixShapedOperator{T}
    A::OP
end
export RowGramOperator

function RowGramOperator(A::MatrixShaped)
    op = asoperator(A)
    return RowGramOperator{eltype(op),typeof(op)}(op)
end

"""
    rowgram_factor(op)

Returns a factor `F` with `op == F * F'`, always as a matrix-shaped
operator, with no guarantee of triangular structure, canonical form or
uniqueness - the loose "matrix square root" `op^(1/2)` of the
sampling/coloring convention in the statistics literature.

Use [`lower_cholesky`](@ref) if canonical triangular structure is
required.

For a [`RowGramOperator`](@ref) this is the stored factor, returned
without computation. [`WoodburyOperator`](@ref)s compute their stable
factorization; wrapped positive-definite matrices, uniform scalings
and structured semi-definite operators (diagonal, zero,
block-diagonal) fall back to [`lower_cholesky`](@ref).
"""
rowgram_factor(op::RowGramOperator) = op.A
export rowgram_factor

Base.size(op::RowGramOperator) = (size(op.A, 1), size(op.A, 1))

Base.adjoint(op::RowGramOperator) = op

Symmetricity(::RowGramOperator{T}) where T =
    NumberDomain(T) === Real ? IsSymmetric() : UnknownSymmetricity()
Hermitianity(::RowGramOperator) = IsHermitian()

PosDefNess(op::RowGramOperator) = _gram_posdefness(RowRankNess(rowgram_factor(op)))
_gram_posdefness(::IsFullRowRank) = IsPosDef()
_gram_posdefness(::RowRankNess) = IsPosSemiDefOnly()

# The stored factor is always an operator and argument shapes were
# checked by the outer entry point, so component application stays in
# the unchecked implementation layer:
explicit_mul_impl(op::RowGramOperator, v::AbstractVector{<:Number}) =
    explicit_mul_impl(op.A, explicit_mul_impl(adjoint(op.A), v))
batched_mul_impl(op::RowGramOperator, X::AbstractMatrix{<:Number}) =
    explicit_mul_impl(op.A, explicit_mul_impl(adjoint(op.A), X))

# Value refinement: a square, structurally triangular factor with
# non-zero diagonal has full rank, making the Gram positive definite:
function LinearAlgebra.isposdef(op::RowGramOperator)
    PosDefNess(op) isa IsPosDef && return true
    return _triangular_fullrank(rowgram_factor(op))
end

_triangular_fullrank(::Any) = false

function _triangular_fullrank(A::AbstractMatrix{<:Number})
    size(A, 1) == size(A, 2) || return false
    Triangularity(A) isa UnknownTriangularity && return false
    return all(!iszero, diag(A))
end

function _triangular_fullrank(op::MatrixAsOperator{<:Number,<:Tuple,<:AbstractMatrix})
    size(op, 1) == size(op, 2) || return false
    Triangularity(op) isa UnknownTriangularity && return false
    return all(!iszero, diag(parent(op)))
end

# det(A * A') == |det(A)|² for a square factor; a singular factor
# propagates the zero sign:
function LinearAlgebra.logabsdet(op::RowGramOperator)
    A = rowgram_factor(op)
    size(A, 1) == size(A, 2) || throw(ArgumentError(
        "logabsdet of a RowGramOperator requires a square Gram factor"
    ))
    la = logabsdet(A)
    return 2 * first(la), abs2(last(la))
end

Base.:(==)(a::RowGramOperator, b::RowGramOperator) = a.A == b.A

Base.hash(op::RowGramOperator, h::UInt) = hash(op.A, hash(:RowGramOperator, h))

Base.isapprox(a::RowGramOperator, b::RowGramOperator; kwargs...) = isapprox(a.A, b.A; kwargs...)

function Base.show(io::IO, op::RowGramOperator)
    print(io, "rowgram_operator(")
    show(io, op.A)
    print(io, ")")
end


# (A A')⁻¹ x solves with the factor, then its adjoint:
function ldiv_impl(op::RowGramOperator, x::AbstractVecOrMat{<:Number})
    A = op.A
    size(A, 1) == size(A, 2) || throw(DimensionMismatch(
        "inverse application of a RowGramOperator requires a square Gram factor"
    ))
    return ldiv_impl(adjoint(A), ldiv_impl(A, x))
end

# A Gram factor that is lower triangular by construction is its own
# lower Cholesky factor, up to canonicalization:
function lower_cholesky(op::RowGramOperator)
    A = rowgram_factor(op)
    Triangularity(A) isa IsLowerTriangularOrDiagonal || throw(ArgumentError(
        "the stored Gram factor is not structurally triangular, no structural lower_cholesky available"
    ))
    return _canonical_chol(A)
end

# Canonicalization by column sign resp. phase factors. The fast-path
# check is a host-side value branch, factors from actual Cholesky
# factorizations are already canonical; zero diagonal entries (singular
# psd factors) keep their column:
_colphase(x) = iszero(x) ? one(x) : conj(sign(x))

function _canonical_chol(A::AbstractMatrix)
    d = diag(A)
    all(x -> isreal(x) && real(x) >= 0, d) && return A
    B = A * Diagonal(map(_colphase, d))
    # rounding in the phase products can leave an O(eps) imaginary
    # residue; the canonical diagonal is exactly real non-negative:
    B[diagind(B)] = abs.(d)
    return B
end

_canonical_chol(op::MatrixAsOperator) = asoperator(_canonical_chol(parent(op)))

# The documented fallback for non-Gram operators is lower_cholesky,
# defined per operator type; without one there is no Gram factor:
rowgram_factor(op::MatrixShapedOperator) = throw(ArgumentError(
    "no structural Gram factor available for operator of type $(nameof(typeof(op)))"
))

