# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).


"""
    struct IdentityOperator <: MatrixShapedOperator{Bool}

Represents the identity operator of an explicit size.

User code should not call `IdentityOperator` directly, but use
[`𝟙`](@ref) instead:

```julia
id = 𝟙(size(x, 1))
id * x == x
```

The element type is the multiplicative neutral `Bool`, which promotes
away; scaled identities are [`UniformScalingOperator`](@ref)s.
"""
struct IdentityOperator <: MatrixShapedOperator{Bool}
    n::Int

    function IdentityOperator(n::Integer)
        n >= 0 || throw(ArgumentError("IdentityOperator size must be non-negative, got $n"))
        return new(Int(n))
    end
end
export IdentityOperator

Base.size(op::IdentityOperator) = (op.n, op.n)

Base.adjoint(op::IdentityOperator) = op

Symmetricity(::IdentityOperator) = IsSymmetric()
Hermitianity(::IdentityOperator) = IsHermitian()
PosDefNess(::IdentityOperator) = IsPosDef()
Triangularity(::IdentityOperator) = IsDiagonal()
Unitarity(::IdentityOperator) = IsUnitary()
RowRankNess(::IdentityOperator) = IsFullRowRank()

# Application and solves are no-ops; results may alias the argument:
explicit_mul_impl(::IdentityOperator, v::AbstractVector{<:Number}) = v
batched_mul_impl(::IdentityOperator, X::AbstractMatrix{<:Number}) = X
ldiv_impl(::IdentityOperator, x::AbstractVecOrMat{<:Number}) = x

lower_cholesky(op::IdentityOperator) = op
rowgram_factor(op::IdentityOperator) = op
_canonical_chol(op::IdentityOperator) = op

# Structural solves need no factorization:
LinearAlgebra.factorize(op::IdentityOperator) = op

LinearAlgebra.logabsdet(::IdentityOperator) = (0.0, true)

asmatrix(op::IdentityOperator) = Diagonal(fill(one(Bool), op.n))

# Multiplying with the identity is a type-level no-op:
lazy_mul_impl(a::MatrixShapedOperator, ::IdentityOperator) = a
lazy_mul_impl(::IdentityOperator, b::MatrixShapedOperator) = b
lazy_mul_impl(a::IdentityOperator, ::IdentityOperator) = a
lazy_mul_impl(a::MatrixShapedProduct, ::IdentityOperator) = a
lazy_mul_impl(::IdentityOperator, b::MatrixShapedProduct) = b

Base.:(==)(a::IdentityOperator, b::IdentityOperator) = a.n == b.n

Base.isapprox(a::IdentityOperator, b::IdentityOperator; kwargs...) = a == b

Base.hash(op::IdentityOperator, h::UInt) = hash(op.n, hash(:IdentityOperator, h))

function Base.show(io::IO, op::IdentityOperator)
    print(io, "𝟙(")
    show(io, op.n)
    print(io, ")")
end


"""
    const UniformScalingOperator{T,S} = ScaledOperator{T,S,IdentityOperator}

A scaled identity `λ * 𝟙(n)`: a type alias for dispatch on uniform
scalings.

User code should not construct `UniformScalingOperator` directly, but
use `λ * 𝟙(n)`, or add a sizeless `λ * 𝟙` resp. `λ * I` to an operator.

In operator products a scaled identity melts into a scalar scaling of
the other factor.
"""
const UniformScalingOperator{T,S} = ScaledOperator{T,S,IdentityOperator}
export UniformScalingOperator

asmatrix(op::UniformScalingOperator) = Diagonal(fill(op.s, op.op.n))

# Structural solves need no factorization:
LinearAlgebra.factorize(op::UniformScalingOperator) = op

# Solving against the identity resp. a scaled identity is exact:
function Base.:(\)(a::IdentityOperator, b::MatrixShapedOperator)
    size(a, 2) == size(b, 1) || throw(DimensionMismatch(
        "operator of size $(size(a)) can't be inverse-applied to operator of size $(size(b))"
    ))
    return b
end
function Base.:(\)(a::UniformScalingOperator, b::MatrixShapedOperator)
    size(a, 1) == 0 && return a.op \ b
    iszero(a.s) && throw(SingularException(1))
    return inv(a.s) * (a.op \ b)
end

# A scaled identity factor melts into a scalar scaling of the other
# factor:
lazy_mul_impl(a::UniformScalingOperator, b::MatrixShapedOperator) = lazy_mul_impl(a.s, b)
lazy_mul_impl(a::MatrixShapedOperator, b::UniformScalingOperator) = lazy_mul_impl(b.s, a)
lazy_mul_impl(a::UniformScalingOperator, b::UniformScalingOperator) = lazy_mul_impl(a.s, b)
lazy_mul_impl(a::UniformScalingOperator, b::MatrixShapedProduct) = lazy_mul_impl(a.s, b)
lazy_mul_impl(a::MatrixShapedProduct, b::UniformScalingOperator) = lazy_mul_impl(b.s, a)
lazy_mul_impl(a::UniformScalingOperator, ::IdentityOperator) = a
lazy_mul_impl(::IdentityOperator, b::UniformScalingOperator) = b


"""
    struct MatrixShapedOperators.One

The type of [`𝟙`](@ref).
"""
struct One end
@compat public One

"""
    const 𝟙 = MatrixShapedOperators.One()

The sizeless identity: a special algebra object (not a `MatrixShapedOperator`)
that takes its size from context.

Similar to `LinearAlgebra.I` but without a scaling factor. [`One`](@ref)
is a singleton type.

`𝟙(n)` constructs the sized identity:

```julia
𝟙(size(x, 1)) * x == x

op + 𝟙 == op + 𝟙(size(op, 1))  # op must be square
op * 𝟙 == op
```
"""
const 𝟙 = One()
export 𝟙

(::One)(n::Integer) = IdentityOperator(n)

Base.:(+)(a::MatrixShapedOperator, ::One) = a + IdentityOperator(size(a, 1))
Base.:(+)(::One, b::MatrixShapedOperator) = IdentityOperator(size(b, 1)) + b
Base.:(-)(a::MatrixShapedOperator, ::One) = a + (-IdentityOperator(size(a, 1)))
Base.:(-)(::One, b::MatrixShapedOperator) = IdentityOperator(size(b, 1)) + (-b)

Base.:(*)(a::MatrixShapedOperator, ::One) = a
Base.:(*)(::One, b::MatrixShapedOperator) = b

Base.:(+)(A::AbstractMatrix{<:Number}, ::One) = A + I
Base.:(+)(::One, A::AbstractMatrix{<:Number}) = I + A
Base.:(-)(A::AbstractMatrix{<:Number}, ::One) = A - I
Base.:(-)(::One, A::AbstractMatrix{<:Number}) = I - A
Base.:(*)(A::AbstractMatrix{<:Number}, ::One) = A * I
Base.:(*)(::One, A::AbstractMatrix{<:Number}) = I * A

Base.:(*)(s::Number, ::One) = s * I
Base.:(*)(::One, s::Number) = I * s
Base.:(/)(::One, s::Number) = I / s

Base.:(*)(::One, v::AbstractVector{<:Number}) = I * v

Base.:(+)(J::UniformScaling, ::One) = J + I
Base.:(+)(::One, J::UniformScaling) = I + J
Base.:(-)(J::UniformScaling, ::One) = J - I
Base.:(-)(::One, J::UniformScaling) = I - J
Base.:(*)(J::UniformScaling, ::One) = J * I
Base.:(*)(::One, J::UniformScaling) = I * J

Base.:(+)(::One, ::One) = I + I
Base.:(-)(::One, ::One) = I - I
Base.:(*)(o::One, ::One) = o
Base.:(-)(::One) = -I

Base.adjoint(o::One) = o
Base.transpose(o::One) = o
Base.conj(o::One) = o

Symmetricity(::One) = IsSymmetric()
Hermitianity(::One) = IsHermitian()
PosDefNess(::One) = IsPosDef()
Triangularity(::One) = IsDiagonal()
Unitarity(::One) = IsUnitary()
RowRankNess(::One) = IsFullRowRank()

lower_cholesky(o::One) = o
rowgram_factor(o::One) = o
colgram_factor(o::One) = o

LinearAlgebra.logabsdet(::One) = (0.0, true)

Base.:(\)(::One, x::AbstractVecOrMat{<:Number}) = x
Base.:(\)(::One, op::MatrixShapedOperator) = op

LinearAlgebra.issymmetric(::One) = true
LinearAlgebra.ishermitian(::One) = true
LinearAlgebra.isposdef(::One) = true
_ispossemidef(::One) = true

function Base.show(io::IO, ::One)
    print(io, "𝟙")
end
