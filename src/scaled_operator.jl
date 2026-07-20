# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).


"""
    struct ScaledOperator{T<:Number,S<:Number,O} <: MatrixShapedOperator{T}

Represents `s * op` for a scalar `s` and a matrix-shaped operator `op`.

User code should not call `ScaledOperator` directly, but use scalar
multiplication:

```julia
sop = s * op
sop * x == s .* (op * x)
```

Scaling drops definiteness, unitarity and rank structure traits, and
hermitianity for complex scalars; `is*` queries refine by the value of
`s`. Nested scalings collapse into a single `ScaledOperator`;
`logabsdet`, inverse application and (for real non-negative scalings)
Cholesky and Gram factors derive from the wrapped operator.
"""
struct ScaledOperator{T<:Number,S<:Number,O<:MatrixShapedOperator} <: MatrixShapedOperator{T}
    s::S
    op::O
end
export ScaledOperator

function ScaledOperator(s::Number, op::MatrixShapedOperator)
    T = promote_type(typeof(s), eltype(op))
    return ScaledOperator{T,typeof(s),typeof(op)}(s, op)
end

lazy_mul_impl(s::Number, op::MatrixShapedOperator) = ScaledOperator(s, op)
lazy_mul_impl(s::Number, op::ScaledOperator) = ScaledOperator(s * op.s, op.op)

Base.size(op::ScaledOperator) = size(op.op)

Base.adjoint(op::ScaledOperator) = ScaledOperator(conj(op.s), adjoint(op.op))

# Conjugation distributes over the scaling, so scaled identities stay
# in the scaled-identity form with its collapse rules:
Base.conj(op::ScaledOperator) = ScaledOperator(conj(op.s), conj(op.op))

function traitsof(op::ScaledOperator{T,S}) where {T,S}
    ts = traitset_scale(traitsof(op.op))
    return NumberDomain(S) === Real ? ts : filter(t -> !(t isa Hermitianity), ts)
end

for C in (:Symmetricity, :Hermitianity, :PosDefNess, :Triangularity, :Unitarity, :RowRankNess)
    @eval $C(op::ScaledOperator) = gettrait(traitsof(op), $C)
end

BatchedMulStyle(op::ScaledOperator) = BatchedMulStyle(op.op)

explicit_mul_impl(op::ScaledOperator, v::AbstractVector{<:Number}) =
    op.s .* explicit_mul_impl(op.op, v)
batched_mul_impl(op::ScaledOperator, X::AbstractMatrix{<:Number}) =
    op.s .* explicit_mul_impl(op.op, X)

# A zero scaling of a non-empty operator is singular and must throw
# like a zero matrix would, not divide by zero; the check is a
# host-side value branch:
function ldiv_impl(op::ScaledOperator, x::AbstractVecOrMat{<:Number})
    iszero(op.s) && size(op, 1) != 0 && throw(SingularException(1))
    return ldiv_impl(op.op, x) ./ op.s
end

# sqrt(s) times a factor of the wrapped operator is a valid factor for
# real non-negative scalings; the check is a host-side value branch:

function lower_cholesky(op::ScaledOperator)
    _check_factor_scaling(op.s, "lower_cholesky")
    return sqrt(real(op.s)) * lower_cholesky(op.op)
end

function rowgram_factor(op::ScaledOperator)
    _check_factor_scaling(op.s, "rowgram_factor")
    return sqrt(real(op.s)) * rowgram_factor(op.op)
end

_check_factor_scaling(s::Number, what::String) =
    isreal(s) && real(s) >= 0 || throw(ArgumentError(
        "$what of a ScaledOperator requires a real non-negative scaling"
    ))

# Canonicalization by column sign resp. phase factors absorbs the
# phase of the scalar, leaving its absolute value:
_canonical_chol(op::ScaledOperator) = abs(op.s) * _canonical_chol(op.op)

# Value refinements by the sign resp. realness of the scalar:
LinearAlgebra.issymmetric(op::ScaledOperator) = issymmetric(op.op)
LinearAlgebra.ishermitian(op::ScaledOperator) = isreal(op.s) && ishermitian(op.op)
LinearAlgebra.isposdef(op::ScaledOperator) = isreal(op.s) && real(op.s) > 0 && isposdef(op.op)
_ispossemidef(op::ScaledOperator) = isreal(op.s) && real(op.s) >= 0 && _ispossemidef(op.op)

# det(s * A) == s^n * det(A); `n * log(abs(s))` would be NaN for a
# zero scaling of an empty operator, whose determinant is one:
function LinearAlgebra.logabsdet(op::ScaledOperator)
    la = logabsdet(op.op)
    n = size(op, 1)
    ls = log(abs(op.s))
    return (iszero(n) ? zero(ls) : n * ls) + first(la), sign(op.s)^n * last(la)
end

Base.:(==)(a::ScaledOperator, b::ScaledOperator) = a.s == b.s && a.op == b.op

Base.hash(op::ScaledOperator, h::UInt) = hash(op.op, hash(op.s, hash(:ScaledOperator, h)))

Base.isapprox(a::ScaledOperator, b::ScaledOperator; kwargs...) =
    isapprox(a.s, b.s; kwargs...) && isapprox(a.op, b.op; kwargs...)

function Base.show(io::IO, op::ScaledOperator)
    _show_scalar(io, op.s)
    print(io, " * ")
    show(io, op.op)
end

_show_scalar(io::IO, s::Real) =
    s < 0 ? (print(io, "("); show(io, s); print(io, ")")) : show(io, s)

function _show_scalar(io::IO, s::Number)
    print(io, "(")
    show(io, s)
    print(io, ")")
end


# Lazy conjugation; all structure traits are preserved under
# conjugation, `transpose(op) == conj(adjoint(op))` for complex
# element types:

struct _ConjOperator{T<:Number,O<:MatrixShapedOperator} <: MatrixShapedOperator{T}
    op::O
end

_ConjOperator(op::MatrixShapedOperator{T}) where T = _ConjOperator{T,typeof(op)}(op)

Base.conj(op::MatrixShapedOperator{T}) where T =
    NumberDomain(T) === Real ? op : _ConjOperator(op)
Base.conj(op::_ConjOperator) = op.op

Base.size(op::_ConjOperator) = size(op.op)

Base.adjoint(op::_ConjOperator) = _ConjOperator(adjoint(op.op))

traitsof(op::_ConjOperator) = traitsof(op.op)

for C in (:Symmetricity, :Hermitianity, :PosDefNess, :Triangularity, :Unitarity, :RowRankNess)
    @eval $C(op::_ConjOperator) = $C(op.op)
end

BatchedMulStyle(op::_ConjOperator) = BatchedMulStyle(op.op)

explicit_mul_impl(op::_ConjOperator, v::AbstractVector{<:Number}) =
    conj(explicit_mul_impl(op.op, conj(v)))
batched_mul_impl(op::_ConjOperator, X::AbstractMatrix{<:Number}) =
    conj(explicit_mul_impl(op.op, conj(X)))

ldiv_impl(op::_ConjOperator, x::AbstractVecOrMat{<:Number}) =
    conj(ldiv_impl(op.op, conj(x)))

LinearAlgebra.issymmetric(op::_ConjOperator) = issymmetric(op.op)
LinearAlgebra.ishermitian(op::_ConjOperator) = ishermitian(op.op)
LinearAlgebra.isposdef(op::_ConjOperator) = isposdef(op.op)
_ispossemidef(op::_ConjOperator) = _ispossemidef(op.op)

# |det(conj(A))| == |det(A)|, the determinant sign conjugates:
function LinearAlgebra.logabsdet(op::_ConjOperator)
    la = logabsdet(op.op)
    return first(la), conj(last(la))
end

# Conjugating a canonical factor keeps the real non-negative diagonal:
_canonical_chol(op::_ConjOperator) = conj(_canonical_chol(op.op))

Base.:(==)(a::_ConjOperator, b::_ConjOperator) = a.op == b.op

Base.hash(op::_ConjOperator, h::UInt) = hash(op.op, hash(:_ConjOperator, h))

Base.isapprox(a::_ConjOperator, b::_ConjOperator; kwargs...) = isapprox(a.op, b.op; kwargs...)

function Base.show(io::IO, op::_ConjOperator)
    print(io, "conj(")
    show(io, op.op)
    print(io, ")")
end
