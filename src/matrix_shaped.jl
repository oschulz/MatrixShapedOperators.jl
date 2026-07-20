# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).


"""
    abstract type MatrixShapedOperator{T<:Number}

Abstract supertype for linear operators with matrix shape and semantics,
accessed through application and operator algebra, not element access.

Operators combine lazily with operators and eagerly with arrays, on
which they act just like a matrix:

```julia
op_a * op_b    # lazy MatrixShapedProduct
op_a + op_b    # lazy MatrixShapedSum
op * x         # eager application to a vector
op * X         # eager application to the columns of a matrix
op \\ x         # eager structural solve
```

Application is out-of-place, though identity-like operations may
return the argument itself. Matrices enter the lazy algebra via
[`asoperator`](@ref); adding a matrix to an operator is an error, so
implicit operators are never materialized by accident.

# Implementation

Subtypes must implement

* `MatrixShapedOperators.explicit_mul_impl(op, x::AbstractVector{<:Number})`
* `Base.adjoint(op)` and `Base.size(op)`

and declare their structural properties via the trait constructors
(see [`MatrixShapedOperators.StructureTrait`](@ref)). Symmetry is
typically only claimed for real element types, where `transpose` equals
`adjoint`. The `is*` queries default to the traits and refine them by
cheap value checks where those exist. [`batched_mul_impl`](@ref) may
specialize batched application, the fallback applies columns
individually (see [`BatchedMulStyle`](@ref)).

`*`, `+` and `\\` are defined centrally and dispatch to
[`lazy_mul_impl`](@ref), [`lazy_add_impl`](@ref),
[`explicit_mul_impl`](@ref) resp. [`ldiv_impl`](@ref) after argument
checking. `transpose`, scalar scaling, `-`, `/`, `mul!` and
`AbstractMatrix`/`Matrix` materialization are generic.
"""
abstract type MatrixShapedOperator{T<:Number} end
export MatrixShapedOperator

"""
    MatrixShaped{T<:Number} = Union{AbstractMatrix{T},MatrixShapedOperator{T}}

Objects with matrix shape and algebra: matrices and matrix-shaped
operators. [`asoperator`](@ref) and [`asmatrix`](@ref) normalize between
the two.

# Extended help

Covers the foreign `AbstractMatrix`, so methods of `Base`/`LinearAlgebra`
functions must not dispatch on it (type piracy) - it is meant for the
arguments of package-owned functions.
"""
const MatrixShaped{T<:Number} = Union{AbstractMatrix{T},MatrixShapedOperator{T}}
export MatrixShaped


Base.eltype(::Type{<:MatrixShapedOperator{T}}) where T = T

Base.ndims(::MatrixShapedOperator) = 2
Base.ndims(::Type{<:MatrixShapedOperator}) = 2

function Base.size(op::MatrixShapedOperator, d::Integer)
    d >= 1 || throw(ArgumentError("dimension out of range, got $d"))
    return d <= 2 ? size(op)[d] : 1
end


"""
    MatrixShapedOperators.explicit_mul_impl(a, b)

Eager multiplication: application of an operator to a vector, or to the
columns of a matrix as a batch, used by `op * x`/`op * X` - in contrast
to the lazy operator-operator `*`.

Subtypes of [`MatrixShapedOperator`](@ref) implement
`explicit_mul_impl(op, x::AbstractVector{<:Number})`. Matrix
application dispatches on [`BatchedMulStyle`](@ref) to
[`batched_mul_impl`](@ref) or to a generic column-wise fallback.
Unchecked - sizes are verified centrally by `op * x` - so do not call
`*_impl` methods directly.
"""
function explicit_mul_impl end
@compat public explicit_mul_impl

"""
    MatrixShapedOperators.batched_mul_impl(op, X::AbstractMatrix)

Native batched application of `op` to the columns of `X`, implemented
by operator types with [`BatchedMul`](@ref) style. Unchecked, do not
call directly.
"""
function batched_mul_impl end
@compat public batched_mul_impl

# The generic matrix method of explicit_mul_impl dispatches on
# BatchedMulStyle, see operator_traits.jl.

"""
    MatrixShapedOperators.lazy_mul_impl(a, b)

Lazy multiplication: combines operators into a
[`MatrixShapedProduct`](@ref) by default, scalar scaling stays an
operator. `Base.:(*)` involving matrix-shaped operators dispatches to
`lazy_mul_impl` after argument checking, subtypes specialize `lazy_mul_impl`
instead, e.g. for eager collapses.
"""
function lazy_mul_impl end
@compat public lazy_mul_impl

"""
    MatrixShapedOperators.lazy_add_impl(a, b)

Lazy addition, a [`MatrixShapedSum`](@ref) by default. `Base.:(+)`
involving matrix-shaped operators dispatches to `lazy_add_impl`, subtypes
specialize `lazy_add_impl` instead of `Base.:(+)`.
"""
function lazy_add_impl end
@compat public lazy_add_impl


"""
    asoperator(A::MatrixShaped, decls::MatrixShapedOperators.StructureTrait...)::MatrixShapedOperator

Returns `A` if it already is a [`MatrixShapedOperator`](@ref), wraps an
`AbstractMatrix` in a [`MatrixAsOperator`](@ref).

Additional trait values declare structural knowledge beyond what the
matrix type expresses, e.g. `asoperator(Σ, IsPosDef())` for a dense
covariance matrix known to be positive definite. Declarations are
trusted, not verified; they must stay valid if the wrapped matrix is
mutated.
"""
asoperator(op::MatrixShapedOperator) = op
export asoperator


"""
    asmatrix(A::MatrixShaped)::AbstractMatrix

Returns `A` itself if it is an `AbstractMatrix`, otherwise a cheap,
backend-preserving matrix representation of the operator `A` (e.g. the
wrapped matrix of a [`MatrixAsOperator`](@ref)), which may alias `A`.

Idempotent dual of [`asoperator`](@ref). Not defined for implicit
operators - materialize those explicitly via `AbstractMatrix(op)`
(the operator's own storage backend where implemented, dense CPU by
default) or `Matrix(op)` (always dense CPU).
"""
asmatrix(A::AbstractMatrix{<:Number}) = A
export asmatrix


# Operator application is defined centrally and dispatches to the
# implementation methods after argument checking. Arrays are data:
# multiplication applies the operator eagerly, to a vector or to the
# columns of a matrix as a batch:

function Base.:(*)(op::MatrixShapedOperator, v::AbstractVector{<:Number})
    size(v, 1) == size(op, 2) || throw(DimensionMismatch(
        "operator of size $(size(op)) can't be applied to vector of length $(length(v))"
    ))
    return explicit_mul_impl(op, v)
end

function Base.:(*)(op::MatrixShapedOperator{T}, X::AbstractMatrix{<:Number}) where T
    size(X, 1) == size(op, 2) || throw(DimensionMismatch(
        "operator of size $(size(op)) can't be applied to matrix of size $(size(X))"
    ))
    size(X, 2) == 0 && return similar(X, promote_type(T, eltype(X)), size(op, 1), 0)
    return explicit_mul_impl(op, X)
end

Base.:(*)(A::AbstractMatrix{<:Number}, op::MatrixShapedOperator) = adjoint(adjoint(op) * adjoint(A))


# Lazy operator algebra, defined centrally as well; matrices enter the
# algebra explicitly via `asoperator`:

function Base.:(*)(a::MatrixShapedOperator, b::MatrixShapedOperator)
    size(a, 2) == size(b, 1) || throw(DimensionMismatch(
        "operator of size $(size(a)) can't be multiplied with operator of size $(size(b))"
    ))
    return lazy_mul_impl(a, b)
end

# Scalar scaling involves no shapes to check:
Base.:(*)(s::Number, op::MatrixShapedOperator) = lazy_mul_impl(s, op)
Base.:(*)(op::MatrixShapedOperator, s::Number) = lazy_mul_impl(s, op)

# Uniform scalings reduce to scalar scaling under multiplication:
Base.:(*)(J::UniformScaling, op::MatrixShapedOperator) = lazy_mul_impl(J.λ, op)
Base.:(*)(op::MatrixShapedOperator, J::UniformScaling) = lazy_mul_impl(J.λ, op)

function Base.:(+)(a::MatrixShapedOperator, b::MatrixShapedOperator)
    size(a) == size(b) || throw(DimensionMismatch(
        "operator of size $(size(a)) can't be added to operator of size $(size(b))"
    ))
    return lazy_add_impl(a, b)
end

# Adding a raw matrix would have to materialize implicit operators, so
# it must be requested explicitly - via asoperator (lazy algebra) or
# AbstractMatrix/Matrix (eager materialization):
function _no_raw_matrix_add()
    throw(ArgumentError(
        "adding or subtracting a matrix and a matrix-shaped operator is not defined: wrap the matrix via asoperator for lazy operator algebra, or materialize the operator explicitly"
    ))
end
Base.:(+)(::MatrixShapedOperator, ::AbstractMatrix{<:Number}) = _no_raw_matrix_add()
Base.:(+)(::AbstractMatrix{<:Number}, ::MatrixShapedOperator) = _no_raw_matrix_add()

# AbstractQ operands are implicit operators themselves (lazy
# reflector-based application, not element storage), so they
# participate in the lazy algebra like operators, not like data:
Base.:(*)(a::MatrixShapedOperator, Q::LinearAlgebra.AbstractQ{<:Number}) = a * asoperator(Q)
Base.:(*)(Q::LinearAlgebra.AbstractQ{<:Number}, b::MatrixShapedOperator) = asoperator(Q) * b
Base.:(+)(a::MatrixShapedOperator, Q::LinearAlgebra.AbstractQ{<:Number}) = a + asoperator(Q)
Base.:(+)(Q::LinearAlgebra.AbstractQ{<:Number}, b::MatrixShapedOperator) = asoperator(Q) + b
Base.:(-)(a::MatrixShapedOperator, Q::LinearAlgebra.AbstractQ{<:Number}) = a - asoperator(Q)
Base.:(-)(Q::LinearAlgebra.AbstractQ{<:Number}, b::MatrixShapedOperator) = asoperator(Q) - b

# Negation, subtraction and scalar division reduce to scaling and
# addition, definiteness is dropped by the scaling trait rules:

Base.:(-)(op::MatrixShapedOperator{T}) where T = lazy_mul_impl(-one(real(T)), op)

Base.:(-)(a::MatrixShapedOperator, b::MatrixShapedOperator) = a + (-b)
Base.:(-)(::MatrixShapedOperator, ::AbstractMatrix{<:Number}) = _no_raw_matrix_add()
Base.:(-)(::AbstractMatrix{<:Number}, ::MatrixShapedOperator) = _no_raw_matrix_add()
Base.:(-)(a::MatrixShapedOperator, J::UniformScaling) = a + (-J.λ) * I
Base.:(-)(J::UniformScaling, b::MatrixShapedOperator) = J.λ * IdentityOperator(size(b, 1)) + (-b)

Base.:(/)(op::MatrixShapedOperator, s::Number) = lazy_mul_impl(inv(s), op)
Base.:(/)(op::MatrixShapedOperator, J::UniformScaling) = lazy_mul_impl(inv(J.λ), op)
Base.:(\)(J::UniformScaling, op::MatrixShapedOperator) = lazy_mul_impl(inv(J.λ), op)


# For real element types transpose and adjoint coincide, for complex
# ones the transpose is the conjugate of the adjoint (`Base.conj` on
# operators is lazy, see mulfunc_operator.jl):
Base.transpose(op::MatrixShapedOperator{T}) where T =
    NumberDomain(T) === Real ? adjoint(op) : conj(adjoint(op))

function Base.:(*)(v_l::LinearAlgebra.Adjoint{<:Number,<:AbstractVector{<:Number}}, op::MatrixShapedOperator)
    return adjoint(adjoint(op) * adjoint(v_l))
end

function Base.:(*)(v_l::LinearAlgebra.Transpose{<:Number,<:AbstractVector{<:Number}}, op::MatrixShapedOperator)
    return transpose(transpose(op) * transpose(v_l))
end


"""
    MatrixShapedOperators.ldiv_impl(op, x::AbstractVecOrMat{<:Number})

Eager inverse application of a square operator to a vector or to the
columns of a matrix, the implementation hook behind `op \\ x`.
Unchecked, do not call directly.
"""
function ldiv_impl end
@compat public ldiv_impl

# Unitary operators invert via their adjoint; there is no generic
# inverse application beyond that:
function ldiv_impl(op::MatrixShapedOperator, x::AbstractVecOrMat{<:Number})
    Unitarity(op) isa IsUnitary && return explicit_mul_impl(adjoint(op), x)
    throw(ArgumentError(
        "no structural inverse application available for operator of type $(nameof(typeof(op)))"
    ))
end

"""
    op::MatrixShapedOperator \\ x::AbstractVector
    op::MatrixShapedOperator \\ X::AbstractMatrix

Eager inverse application (solve) of a square operator to a vector, or
to the columns of a matrix as multiple right-hand sides.

# Extended help

Like [`lower_cholesky`](@ref) and `logabsdet` this is structural: it
follows the inverse path of the operator's representation, or throws an
`ArgumentError` if there is none. `LinearAlgebra.factorize` caches a
factorization for repeated solves.
"""
function Base.:(\)(op::MatrixShapedOperator{T}, x::AbstractVecOrMat{<:Number}) where T
    size(op, 1) == size(op, 2) || throw(DimensionMismatch(
        "inverse application requires a square operator, got size $(size(op))"
    ))
    size(x, 1) == size(op, 1) || throw(DimensionMismatch(
        "operator of size $(size(op)) can't be inverse-applied to argument of size $(size(x))"
    ))
    x isa AbstractMatrix && size(x, 2) == 0 &&
        return similar(x, promote_type(T, eltype(x)), size(op, 2), 0)
    return ldiv_impl(op, x)
end

# Reserved for lazy inverse application in the operator algebra:
function Base.:(\)(::MatrixShapedOperator, ::MatrixShapedOperator)
    throw(ArgumentError(
        "lazy inverse application between matrix-shaped operators is not available yet"
    ))
end

# Comparison of operators is representational: operators of different
# representation compare unequal, like under `==`:
Base.isapprox(a::MatrixShapedOperator, b::MatrixShapedOperator; kwargs...) = false


# Application of matrix-shaped operators and matrix-like objects alike,
# for implementations that deliberately hold components of either kind
# (e.g. WoodburyOperator); stays in the unchecked implementation layer:
_apply(A, x::AbstractVecOrMat{<:Number}) = A * x
_apply(op::MatrixShapedOperator, x::AbstractVecOrMat{<:Number}) = explicit_mul_impl(op, x)


# An interoperability shim, not a performance interface: computes
# out-of-place via `op * x` and copies into `y`, so it allocates.
function LinearAlgebra.mul!(y::AbstractVecOrMat{<:Number}, op::MatrixShapedOperator, x::AbstractVecOrMat{<:Number})
    return mul!(y, op, x, true, false)
end

function LinearAlgebra.mul!(
    y::AbstractVecOrMat{<:Number}, op::MatrixShapedOperator, x::AbstractVecOrMat{<:Number},
    alpha::Number, beta::Number
)
    w = op * x
    if iszero(beta)
        y .= alpha .* w
    else
        y .= alpha .* w .+ beta .* y
    end
    return y
end

function LinearAlgebra.mul!(Y::AbstractMatrix{<:Number}, A::AbstractMatrix{<:Number}, op::MatrixShapedOperator)
    return mul!(Y, A, op, true, false)
end

function LinearAlgebra.mul!(
    Y::AbstractMatrix{<:Number}, A::AbstractMatrix{<:Number}, op::MatrixShapedOperator,
    alpha::Number, beta::Number
)
    W = A * op
    if iszero(beta)
        Y .= alpha .* W
    else
        Y .= alpha .* W .+ beta .* Y
    end
    return Y
end


# Materialization: AbstractMatrix(op) applies op to an identity matrix
# of the backend given by _one_matrix (CPU by default, wrapper types
# may use their parent's backend), Matrix(op) guarantees dense CPU:

_one_matrix(op::MatrixShapedOperator{T}) where T = Matrix{T}(I, size(op, 2), size(op, 2))

Base.AbstractMatrix(op::MatrixShapedOperator) = op * _one_matrix(op)
Base.Matrix(op::MatrixShapedOperator) = Matrix(AbstractMatrix(op))
Base.convert(::Type{AbstractMatrix}, op::MatrixShapedOperator) = AbstractMatrix(op)
Base.convert(::Type{Matrix}, op::MatrixShapedOperator) = Matrix(op)


