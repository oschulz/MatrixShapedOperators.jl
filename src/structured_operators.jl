# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).


"""
    diagonal_operator(d::AbstractVector{<:Number})

Returns a [`MatrixAsOperator`](@ref) around
`LinearAlgebra.Diagonal(d)`, so that specializations for `Diagonal`
(e.g. on GPU or under program tracing) take effect.
"""
diagonal_operator(d::AbstractVector{<:Number}) = MatrixAsOperator(Diagonal(d))
export diagonal_operator

const _DiagonalAsOperator{T} = MatrixAsOperator{T,<:Tuple,<:Diagonal}


"""
    struct BlockDiagOperator{T<:Number,BS<:Union{Tuple,AbstractVector}} <: MatrixShapedOperator{T}

Represents the block-diagonal composition of matrix-shaped operators.
Blocks need not be square, they may be a tuple (few, heterogeneous) or
a vector (many, uniform).

User code should not call `BlockDiagOperator` directly, but use
[`blockdiag_operator`](@ref) instead.

Traits derive from the blocks via the trait algebra;
[`lower_cholesky`](@ref) and `logabsdet` operate block-wise.
"""
struct BlockDiagOperator{T<:Number,BS<:Union{Tuple,AbstractVector}} <: MatrixShapedOperator{T}
    blocks::BS

    function BlockDiagOperator{T,BS}(blocks::BS) where {T<:Number,BS<:Union{Tuple,AbstractVector}}
        isempty(blocks) && throw(ArgumentError("BlockDiagOperator requires at least one block"))
        return new{T,BS}(blocks)
    end
end
export BlockDiagOperator

function BlockDiagOperator(blocks::Union{Tuple,AbstractVector})
    isempty(blocks) && throw(ArgumentError("BlockDiagOperator requires at least one block"))
    ops = map(asoperator, blocks)
    T = mapreduce(eltype, promote_type, ops)
    return BlockDiagOperator{T,typeof(ops)}(ops)
end

Base.size(op::BlockDiagOperator) =
    (sum(b -> size(b, 1), op.blocks), sum(b -> size(b, 2), op.blocks))

# The trait-set algebra is shape-blind, but with all blocks square the
# block-diagonal inherits their triangularity under the additive join;
# block shapes are static, so this stays trace-safe:
function traitsof(op::BlockDiagOperator)
    ts = mapreduce(traitsof, traitset_blockdiag, op.blocks)
    all(b -> size(b, 1) == size(b, 2), op.blocks) || return ts
    tri = mapreduce(Triangularity, trait_add, op.blocks)
    return tri isa UnknownTriangularity ? ts : traitset(tri, ts...)
end

for C in (:Symmetricity, :Hermitianity, :PosDefNess, :Triangularity, :Unitarity, :RowRankNess)
    @eval $C(op::BlockDiagOperator) = gettrait(traitsof(op), $C)
end

Base.adjoint(op::BlockDiagOperator) = BlockDiagOperator(map(adjoint, op.blocks))

BatchedMulStyle(::BlockDiagOperator) = BatchedMul()

function explicit_mul_impl(op::BlockDiagOperator, v::AbstractVector{<:Number})
    rs = _block_ranges(map(b -> size(b, 2), op.blocks))
    return reduce(vcat, map((b, r) -> explicit_mul_impl(b, v[r]), op.blocks, rs))
end

function batched_mul_impl(op::BlockDiagOperator, X::AbstractMatrix{<:Number})
    rs = _block_ranges(map(b -> size(b, 2), op.blocks))
    return reduce(vcat, map((b, r) -> explicit_mul_impl(b, X[r, :]), op.blocks, rs))
end

# Tuple vs. vector block storage is a performance choice and does not
# affect equality, like for sums and products:
Base.:(==)(a::BlockDiagOperator, b::BlockDiagOperator) = _parts_equal(a.blocks, b.blocks)

Base.hash(op::BlockDiagOperator, h::UInt) = _parts_hash(op.blocks, hash(:BlockDiagOperator, h))

Base.isapprox(a::BlockDiagOperator, b::BlockDiagOperator; kwargs...) =
    length(a.blocks) == length(b.blocks) &&
    all(isapprox(ba, bb; kwargs...) for (ba, bb) in zip(a.blocks, b.blocks))

function Base.show(io::IO, op::BlockDiagOperator)
    print(io, "blockdiag_operator(")
    _show_blocks(io, op.blocks)
    print(io, ")")
end

function _show_blocks(io::IO, blocks::Tuple)
    for (i, b) in enumerate(blocks)
        i > 1 && print(io, ", ")
        show(io, b)
    end
end

_show_blocks(io::IO, blocks::AbstractVector) = show(io, blocks)

function _block_ranges(lens::NTuple{N,Integer}) where N
    stops = cumsum(lens)
    starts = (1, map(s -> s + 1, Base.front(stops))...)
    return map((a, b) -> a:b, starts, stops)
end

function _block_ranges(lens::AbstractVector{<:Integer})
    stops = cumsum(lens)
    return map((l, b) -> (b - l + 1):b, lens, stops)
end

# The determinant of a block-diagonal operator is the product of the
# block determinants:
function LinearAlgebra.logabsdet(op::BlockDiagOperator)
    all(b -> size(b, 1) == size(b, 2), op.blocks) || throw(DimensionMismatch(
        "logabsdet of a block-diagonal operator requires square blocks"
    ))
    parts = map(logabsdet, op.blocks)
    return sum(first, parts), prod(last, parts)
end

# Block-diagonal solves operate block-wise:
function ldiv_impl(op::BlockDiagOperator, x::AbstractVecOrMat{<:Number})
    all(b -> size(b, 1) == size(b, 2), op.blocks) || throw(DimensionMismatch(
        "inverse application of a block-diagonal operator requires square blocks"
    ))
    rs = _block_ranges(map(b -> size(b, 1), op.blocks))
    return reduce(vcat, map((b, r) -> ldiv_impl(b, _block_rows(x, r)), op.blocks, rs))
end

_block_rows(x::AbstractVector, r::AbstractUnitRange) = x[r]
_block_rows(X::AbstractMatrix, r::AbstractUnitRange) = X[r, :]

lower_cholesky(op::BlockDiagOperator) = BlockDiagOperator(map(lower_cholesky, op.blocks))

rowgram_factor(op::BlockDiagOperator) = BlockDiagOperator(map(rowgram_factor, op.blocks))

_canonical_chol(op::BlockDiagOperator) = BlockDiagOperator(map(_canonical_chol, op.blocks))


"""
    blockdiag_operator(blocks::MatrixShaped...)
    blockdiag_operator(blocks::AbstractVector{<:MatrixShaped})

Returns a matrix-shaped operator with the given block-diagonal
structure, a [`BlockDiagOperator`](@ref) in general.

The blocks are [`MatrixShaped`](@ref) values and need not be square,
given either as individual arguments (best for a few blocks of different
type) or as a vector (best for many blocks of equal type).

# Extended help

Diagonal-like blocks (`Diagonal`, [`diagonal_operator`](@ref),
[`UniformScalingOperator`](@ref), the sized identity) collapse into a
single diagonal operator, sparse blocks into a single wrapped sparse
matrix (SparseArrays extension), and [`RowGramOperator`](@ref) resp.
[`ColGramOperator`](@ref) blocks into a single Gram operator of the
block-diagonal of their factors.
"""
function blockdiag_operator end
export blockdiag_operator

blockdiag_operator(b1::MatrixShaped) = asoperator(b1)

blockdiag_operator(b1::MatrixShaped, bs::MatrixShaped...) = BlockDiagOperator((b1, bs...))

# Diagonal-like blocks collapse into a single wrapped Diagonal:

const _ScaledDiagonalLike = ScaledOperator{<:Number,<:Number,<:Union{IdentityOperator,_DiagonalAsOperator}}
const _DiagonalLike = Union{Diagonal{<:Number},_DiagonalAsOperator,_ScaledDiagonalLike,IdentityOperator}

_diag_of(A::Diagonal) = A.diag
_diag_of(op::_DiagonalAsOperator) = parent(op).diag
_diag_of(op::_ScaledDiagonalLike) = op.s .* _diag_of(op.op)
_diag_of(op::IdentityOperator) = fill(one(Bool), op.n)

blockdiag_operator(b1::_DiagonalLike) = asoperator(b1)

function blockdiag_operator(b1::_DiagonalLike, bs::_DiagonalLike...)
    return diagonal_operator(vcat(_diag_of(b1), map(_diag_of, bs)...))
end

blockdiag_operator(b1::RowGramOperator) = b1

function blockdiag_operator(b1::RowGramOperator, bs::RowGramOperator...)
    return rowgram_operator(blockdiag_operator(rowgram_factor(b1), map(rowgram_factor, bs)...))
end

blockdiag_operator(b1::ColGramOperator) = b1

function blockdiag_operator(b1::ColGramOperator, bs::ColGramOperator...)
    return colgram_operator(blockdiag_operator(colgram_factor(b1), map(colgram_factor, bs)...))
end

function blockdiag_operator(blocks::AbstractVector{<:MatrixShaped})
    isempty(blocks) && throw(ArgumentError("blockdiag_operator requires at least one block"))
    length(blocks) == 1 && return asoperator(only(blocks))
    return BlockDiagOperator(blocks)
end

function blockdiag_operator(blocks::AbstractVector{<:_DiagonalLike})
    isempty(blocks) && throw(ArgumentError("blockdiag_operator requires at least one block"))
    return diagonal_operator(reduce(vcat, map(_diag_of, blocks)))
end

function blockdiag_operator(blocks::AbstractVector{<:RowGramOperator})
    isempty(blocks) && throw(ArgumentError("blockdiag_operator requires at least one block"))
    length(blocks) == 1 && return only(blocks)
    return rowgram_operator(blockdiag_operator(map(rowgram_factor, blocks)))
end

function blockdiag_operator(blocks::AbstractVector{<:ColGramOperator})
    isempty(blocks) && throw(ArgumentError("blockdiag_operator requires at least one block"))
    length(blocks) == 1 && return only(blocks)
    return colgram_operator(blockdiag_operator(map(colgram_factor, blocks)))
end
