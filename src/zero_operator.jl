# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).


"""
    struct ZeroOperator <: MatrixShapedOperator{Bool}

Represents the zero operator of an explicit, possibly rectangular
size.

User code should not call `ZeroOperator` directly, but use
[`𝟘`](@ref) instead; `Base.zero` of any matrix-shaped operator also
returns one:

```julia
z = 𝟘(3, 2)
z * x == zeros(3)  # x of length 2
zero(op) isa ZeroOperator
```

Adding a zero operator is a type-level no-op and multiplication with
one collapses to a zero operator of the resulting shape. The element
type is the additive neutral `Bool`, which promotes away in operator
algebra.
"""
struct ZeroOperator <: MatrixShapedOperator{Bool}
    m::Int
    n::Int

    function ZeroOperator(m::Integer, n::Integer)
        m >= 0 && n >= 0 || throw(ArgumentError("ZeroOperator size must be non-negative, got ($m, $n)"))
        return new(Int(m), Int(n))
    end
end
export ZeroOperator

Base.size(op::ZeroOperator) = (op.m, op.n)

Base.zero(op::MatrixShapedOperator) = ZeroOperator(size(op)...)
Base.iszero(::ZeroOperator) = true

Base.adjoint(op::ZeroOperator) = ZeroOperator(op.n, op.m)
Base.conj(op::ZeroOperator) = op

# Square zero operators are structurally symmetric, hermitian,
# positive semi-definite (never definite) and diagonal:
Symmetricity(op::ZeroOperator) = op.m == op.n ? IsSymmetric() : UnknownSymmetricity()
Hermitianity(op::ZeroOperator) = op.m == op.n ? IsHermitian() : UnknownHermitianity()
PosDefNess(op::ZeroOperator) = op.m == op.n ? IsPosSemiDefOnly() : UnknownPosDefNess()
Triangularity(op::ZeroOperator) = op.m == op.n ? IsDiagonal() : UnknownTriangularity()

LinearAlgebra.issymmetric(op::ZeroOperator) = op.m == op.n
LinearAlgebra.ishermitian(op::ZeroOperator) = op.m == op.n
_ispossemidef(op::ZeroOperator) = op.m == op.n

explicit_mul_impl(op::ZeroOperator, v::AbstractVector{<:Number}) =
    fill!(similar(v, promote_type(Bool, eltype(v)), op.m), 0)
batched_mul_impl(op::ZeroOperator, X::AbstractMatrix{<:Number}) =
    fill!(similar(X, promote_type(Bool, eltype(X)), op.m, size(X, 2)), 0)

# The zero operator is singular, except for size zero:
ldiv_impl(op::ZeroOperator, x::AbstractVecOrMat{<:Number}) =
    iszero(op.m) ? x : throw(SingularException(1))

# The square zero operator is its own (singular) factor:
lower_cholesky(op::ZeroOperator) = _zero_factor(op)
rowgram_factor(op::ZeroOperator) = _zero_factor(op)

function _zero_factor(op::ZeroOperator)
    op.m == op.n || throw(DimensionMismatch(
        "factors of a zero operator require a square size, got $(size(op))"
    ))
    return op
end

_canonical_chol(op::ZeroOperator) = _zero_factor(op)

# Solves fail as singular with or without factorization:
LinearAlgebra.factorize(op::ZeroOperator) = op

function LinearAlgebra.logabsdet(op::ZeroOperator)
    op.m == op.n || throw(DimensionMismatch(
        "logabsdet requires a square operator, got size $(size(op))"
    ))
    return iszero(op.m) ? (0.0, true) : (-Inf, false)
end

# Adding a zero operator is a type-level no-op:
lazy_add_impl(a::MatrixShapedOperator, ::ZeroOperator) = a
lazy_add_impl(::ZeroOperator, b::MatrixShapedOperator) = b
lazy_add_impl(a::ZeroOperator, ::ZeroOperator) = a
lazy_add_impl(a::MatrixShapedSum, ::ZeroOperator) = a
lazy_add_impl(::ZeroOperator, b::MatrixShapedSum) = b

# Multiplication with a zero operator collapses to the resulting
# zero shape:
_zero_product(a, b) = ZeroOperator(size(a, 1), size(b, 2))

lazy_mul_impl(a::MatrixShapedOperator, z::ZeroOperator) = _zero_product(a, z)
lazy_mul_impl(z::ZeroOperator, b::MatrixShapedOperator) = _zero_product(z, b)
lazy_mul_impl(a::ZeroOperator, b::ZeroOperator) = _zero_product(a, b)
lazy_mul_impl(a::MatrixShapedProduct, z::ZeroOperator) = _zero_product(a, z)
lazy_mul_impl(z::ZeroOperator, b::MatrixShapedProduct) = _zero_product(z, b)
lazy_mul_impl(a::IdentityOperator, z::ZeroOperator) = _zero_product(a, z)
lazy_mul_impl(z::ZeroOperator, b::IdentityOperator) = _zero_product(z, b)
lazy_mul_impl(a::UniformScalingOperator, z::ZeroOperator) = _zero_product(a, z)
lazy_mul_impl(z::ZeroOperator, b::UniformScalingOperator) = _zero_product(z, b)

# Scalar scaling of the zero operator is absorbed:
lazy_mul_impl(::Number, z::ZeroOperator) = z

Base.:(==)(a::ZeroOperator, b::ZeroOperator) = size(a) == size(b)

Base.hash(op::ZeroOperator, h::UInt) = hash(op.n, hash(op.m, hash(:ZeroOperator, h)))

Base.isapprox(a::ZeroOperator, b::ZeroOperator; kwargs...) = size(a) == size(b)

function Base.show(io::IO, op::ZeroOperator)
    print(io, "𝟘(")
    show(io, op.m)
    print(io, ", ")
    show(io, op.n)
    print(io, ")")
end


"""
    struct MatrixShapedOperators.Zero

The type of [`𝟘`](@ref).
"""
struct Zero end
@compat public Zero

"""
    const 𝟘 = MatrixShapedOperators.Zero()

The sizeless zero: a special algebra object (not a `MatrixShapedOperator`)
that takes its size from context.

[`Zero`](@ref) is a singleton type; `𝟘(m, n)` resp. `𝟘(n)` construct
the sized (square) zero:

```julia
𝟘(3, 2) isa ZeroOperator

op + 𝟘 == op
op * 𝟘 == zero(op)
```
"""
const 𝟘 = Zero()
export 𝟘

(::Zero)(m::Integer, n::Integer) = ZeroOperator(m, n)
(::Zero)(n::Integer) = ZeroOperator(n, n)

Base.:(+)(a::MatrixShapedOperator, ::Zero) = a
Base.:(+)(::Zero, b::MatrixShapedOperator) = b
Base.:(-)(a::MatrixShapedOperator, ::Zero) = a
Base.:(-)(::Zero, b::MatrixShapedOperator) = -b

Base.:(*)(a::MatrixShapedOperator, ::Zero) = zero(a)
Base.:(*)(::Zero, b::MatrixShapedOperator) = zero(b)

Base.:(+)(A::AbstractMatrix{<:Number}, ::Zero) = A + zero(I)
Base.:(+)(::Zero, A::AbstractMatrix{<:Number}) = zero(I) + A
Base.:(-)(A::AbstractMatrix{<:Number}, ::Zero) = A - zero(I)
Base.:(-)(::Zero, A::AbstractMatrix{<:Number}) = zero(I) - A
Base.:(*)(A::AbstractMatrix{<:Number}, ::Zero) = A * zero(I)
Base.:(*)(::Zero, A::AbstractMatrix{<:Number}) = zero(I) * A
Base.:(*)(::Zero, v::AbstractVector{<:Number}) = zero(I) * v

Base.:(*)(s::Number, ::Zero) = s * zero(I)
Base.:(*)(::Zero, s::Number) = zero(I) * s

Base.:(+)(J::UniformScaling, ::Zero) = J + zero(I)
Base.:(+)(::Zero, J::UniformScaling) = zero(I) + J
Base.:(-)(J::UniformScaling, ::Zero) = J - zero(I)
Base.:(-)(::Zero, J::UniformScaling) = zero(I) - J
Base.:(*)(J::UniformScaling, ::Zero) = J * zero(I)
Base.:(*)(::Zero, J::UniformScaling) = zero(I) * J

Base.:(+)(z::Zero, ::Zero) = z
Base.:(-)(z::Zero, ::Zero) = z
Base.:(*)(z::Zero, ::Zero) = z
Base.:(-)(z::Zero) = z

Base.:(+)(::One, ::Zero) = I + zero(I)
Base.:(+)(::Zero, ::One) = zero(I) + I
Base.:(-)(::One, ::Zero) = I - zero(I)
Base.:(-)(::Zero, ::One) = zero(I) - I
Base.:(*)(::One, z::Zero) = z
Base.:(*)(z::Zero, ::One) = z

Base.adjoint(z::Zero) = z
Base.transpose(z::Zero) = z
Base.conj(z::Zero) = z
Base.iszero(::Zero) = true

Symmetricity(::Zero) = IsSymmetric()
Hermitianity(::Zero) = IsHermitian()
PosDefNess(::Zero) = IsPosSemiDefOnly()
Triangularity(::Zero) = IsDiagonal()
Unitarity(::Zero) = UnknownUnitarity()
RowRankNess(::Zero) = UnknownRowRankNess()

lower_cholesky(z::Zero) = z
rowgram_factor(z::Zero) = z
colgram_factor(z::Zero) = z

LinearAlgebra.logabsdet(::Zero) = (-Inf, false)

Base.:(\)(::Zero, x::AbstractVecOrMat{<:Number}) = throw(SingularException(1))
Base.:(\)(::Zero, op::MatrixShapedOperator) = throw(SingularException(1))

LinearAlgebra.issymmetric(::Zero) = true
LinearAlgebra.ishermitian(::Zero) = true
LinearAlgebra.isposdef(::Zero) = false
_ispossemidef(::Zero) = true

function Base.show(io::IO, ::Zero)
    print(io, "𝟘")
end

# A zero base yields a pure low-rank operator `B * D * B'` (useful
# with indefinite `D`; for positive semi-definite `D` prefer
# `rowgram_operator(B * lower_cholesky(D))`):
woodbury_operator(::Zero, B::MatrixShaped, D::MatrixShaped) =
    woodbury_operator(ZeroOperator(size(B, 1), size(B, 1)), B, D)
