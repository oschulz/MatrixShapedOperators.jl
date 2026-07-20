# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).


"""
    struct MatrixAsOperator{T,TRS<:Tuple,M} <: MatrixShapedOperator{T}

Wraps an `AbstractMatrix` or `LinearAlgebra.AbstractQ` as a
[`MatrixShapedOperator`](@ref) `op`.

The wrapped object is available via `Base.parent(op)`.

User code should not call `MatrixAsOperator` directly, but use
[`asoperator`](@ref) instead.

# Extended help

Application delegates to multiplication with the wrapped matrix, so
specialized matrix types (e.g. `Diagonal`, GPU arrays) keep their
optimized code paths.

The structure traits are captured at construction as a
[`traitset`](@ref) in the type parameter `TRS` - from the structure of
the wrapped matrix by default, plus declarations via
[`asoperator`](@ref).
"""
struct MatrixAsOperator{T<:Number,TRS<:Tuple,M<:Union{AbstractMatrix{T},LinearAlgebra.AbstractQ{T}}} <: MatrixShapedOperator{T}
    A::M
end
export MatrixAsOperator

function MatrixAsOperator(A::Union{AbstractMatrix{T},LinearAlgebra.AbstractQ{T}}, decls...) where {T<:Number}
    ts = traitset(decls..., traitsof(A)...)
    all(t -> t isa RowRankNess, ts) || size(A, 1) == size(A, 2) || throw(ArgumentError(
        "MatrixAsOperator of size $(size(A)) can't carry square-only structure declarations"
    ))
    return MatrixAsOperator{T,typeof(ts),typeof(A)}(A)
end

MatrixAsOperator(op::MatrixAsOperator) = op

Base.parent(op::MatrixAsOperator) = op.A

Base.size(op::MatrixAsOperator) = size(op.A)

traitsof(::MatrixAsOperator{T,TRS}) where {T,TRS} = _traits_instance(TRS)

for C in (:Symmetricity, :Hermitianity, :PosDefNess, :Triangularity, :Unitarity, :RowRankNess)
    @eval $C(::MatrixAsOperator{T,TRS}) where {T,TRS} = gettrait(TRS, $C)
end

function Base.adjoint(op::MatrixAsOperator)
    return MatrixAsOperator(adjoint(parent(op)), traitset_adjoint(traitsof(op))...)
end

explicit_mul_impl(op::MatrixAsOperator, v::AbstractVector{<:Number}) = parent(op) * v
batched_mul_impl(op::MatrixAsOperator, X::AbstractMatrix{<:Number}) = parent(op) * X

# Trait declarations are type-level metadata and do not participate in equality,
# approximate equality or hashing:

Base.:(==)(a::MatrixAsOperator, b::MatrixAsOperator) = parent(a) == parent(b)
Base.:(==)(a::MatrixAsOperator{<:Number,<:Tuple,<:AbstractMatrix}, B::AbstractMatrix{<:Number}) = parent(a) == B
Base.:(==)(A::AbstractMatrix{<:Number}, b::MatrixAsOperator{<:Number,<:Tuple,<:AbstractMatrix}) = A == parent(b)

Base.isequal(a::MatrixAsOperator, b::MatrixAsOperator) = isequal(parent(a), parent(b))
Base.isequal(a::MatrixAsOperator{<:Number,<:Tuple,<:AbstractMatrix}, B::AbstractMatrix{<:Number}) = isequal(parent(a), B)
Base.isequal(A::AbstractMatrix{<:Number}, b::MatrixAsOperator{<:Number,<:Tuple,<:AbstractMatrix}) = isequal(A, parent(b))

Base.hash(op::MatrixAsOperator, h::UInt) = _parent_hash(parent(op), h)

_parent_hash(A::AbstractMatrix{<:Number}, h::UInt) = hash(A, h)

# AbstractQ values hash by objectid (the Base default), which would
# break hash consistency with the value-comparing `==`; hash weakly by
# type and size instead:
_parent_hash(Q::LinearAlgebra.AbstractQ{<:Number}, h::UInt) =
    hash(size(Q), hash(nameof(typeof(Q)), hash(:MatrixAsOperatorQ, h)))

Base.isapprox(a::MatrixAsOperator, b::MatrixAsOperator; kwargs...) = isapprox(parent(a), parent(b); kwargs...)
Base.isapprox(a::MatrixAsOperator{<:Number,<:Tuple,<:AbstractMatrix}, B::AbstractMatrix{<:Number}; kwargs...) = isapprox(parent(a), B; kwargs...)
Base.isapprox(A::AbstractMatrix{<:Number}, b::MatrixAsOperator{<:Number,<:Tuple,<:AbstractMatrix}; kwargs...) = isapprox(A, parent(b); kwargs...)

function Base.show(io::IO, op::MatrixAsOperator)
    print(io, "MatrixAsOperator(")
    show(io, parent(op))
    for t in traitsof(op)
        print(io, ", ")
        show(io, t)
    end
    print(io, ")")
end

# AbstractQ wrappers materialize via the generic Q * I application:
Base.AbstractMatrix(op::MatrixAsOperator{<:Number,<:Tuple,<:AbstractMatrix}) = copy(parent(op))

LinearAlgebra.logabsdet(op::MatrixAsOperator{<:Number,<:Tuple,<:AbstractMatrix}) = logabsdet(parent(op))

# |det| of a unitary wrapped Q is one, det yields the sign resp. phase;
# the parent-dispatch helper covers adjoint Qs, whose det LinearAlgebra
# does not define:
function LinearAlgebra.logabsdet(op::MatrixAsOperator{<:Number,<:Tuple,<:LinearAlgebra.AbstractQ})
    d = _q_det(parent(op))
    return log(abs(d)), sign(d)
end

_q_det(Q::LinearAlgebra.AbstractQ) = det(Q)
_q_det(Q::LinearAlgebra.AdjointQ) = conj(det(parent(Q)))


asoperator(A::AbstractMatrix{<:Number}, decls...) = MatrixAsOperator(A, decls...)
asoperator(Q::LinearAlgebra.AbstractQ{<:Number}, decls...) = MatrixAsOperator(Q, decls...)

# Declarations can be added to an already-wrapped matrix, merging with
# the stored traits:
asoperator(op::MatrixAsOperator, d1::StructureTrait, decls::StructureTrait...) =
    MatrixAsOperator(parent(op), d1, decls..., traitsof(op)...)

asmatrix(op::MatrixAsOperator{<:Number,<:Tuple,<:AbstractMatrix}) = parent(op)

# Known hermitianity (declared or structural) replaces the value-level
# check `cholesky` performs on plain matrices - faster, and robust
# against rounding-level asymmetries in computed covariances:
function lower_cholesky(op::MatrixAsOperator{<:Number,<:Tuple,<:AbstractMatrix})
    A = parent(op)
    Ah = Hermitianity(op) isa IsHermitian ? _assume_hermitian(A) : A
    return asoperator(lower_cholesky(Ah))
end

_assume_hermitian(A::AbstractMatrix) = Hermitian(A)
_assume_hermitian(A::Union{Diagonal,Hermitian}) = A
# A complex Symmetric is not hermitian storage - re-wrap its parent:
_assume_hermitian(A::Symmetric{<:Real}) = A
_assume_hermitian(A::Symmetric) = Hermitian(parent(A), Symbol(A.uplo))

# Wrapped-matrix solves delegate to the parent; known positive
# definiteness selects a Cholesky solve where the parent has no
# cheaper structural solve of its own:
function ldiv_impl(op::MatrixAsOperator{<:Number,<:Tuple,<:AbstractMatrix}, x::AbstractVecOrMat{<:Number})
    A = parent(op)
    if PosDefNess(op) isa IsPosDef && Triangularity(op) isa UnknownTriangularity
        return cholesky(_assume_hermitian(A)) \ x
    end
    return A \ x
end

# Wrapped-matrix factorization caches the LinearAlgebra factorization
# for repeated solves, with the same declared-posdef Cholesky selection
# as ldiv_impl; structured parents self-return like under `factorize`:
function LinearAlgebra.factorize(op::MatrixAsOperator{<:Number,<:Tuple,<:AbstractMatrix})
    A = parent(op)
    if PosDefNess(op) isa IsPosDef && Triangularity(op) isa UnknownTriangularity
        return cholesky(_assume_hermitian(A))
    end
    return factorize(A)
end

# Q solves are adjoint applications:
ldiv_impl(op::MatrixAsOperator{<:Number,<:Tuple,<:LinearAlgebra.AbstractQ}, x::AbstractVecOrMat{<:Number}) =
    parent(op)' * x

rowgram_factor(op::MatrixAsOperator{<:Number,<:Tuple,<:AbstractMatrix}) = lower_cholesky(op)
