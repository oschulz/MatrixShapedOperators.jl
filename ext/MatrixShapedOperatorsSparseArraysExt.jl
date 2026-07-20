# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).

module MatrixShapedOperatorsSparseArraysExt

using SparseArrays: SparseArrays, SparseMatrixCSC

import MatrixShapedOperators
using MatrixShapedOperators: MatrixShapedOperator, MatrixAsOperator,
    BlockDiagOperator, UniformScalingOperator, IdentityOperator, ZeroOperator,
    asoperator, asmatrix, blockdiag_operator
using MatrixShapedOperators: traitsof, traitset_blockdiag


# Sparse blocks collapse into a single wrapped sparse matrix - a cheap
# structure concatenation, and the result applies as a single sparse
# matrix-vector product. Dense blocks deliberately stay a lazy
# BlockDiagOperator (better per-block BLAS performance, tracing- and
# GPU-compatible, keeps the block structure for structural operations):

const _SparseAsOperator{T} = MatrixAsOperator{T,<:Tuple,<:SparseMatrixCSC}
const _SparseLike = Union{SparseMatrixCSC{<:Number},_SparseAsOperator}

_sparse_of(A::SparseMatrixCSC) = A
_sparse_of(op::_SparseAsOperator) = parent(op)

MatrixShapedOperators.blockdiag_operator(b1::_SparseLike) = asoperator(b1)

function MatrixShapedOperators.blockdiag_operator(b1::_SparseLike, bs::_SparseLike...)
    ts = mapreduce(traitsof, traitset_blockdiag, (b1, bs...))
    return MatrixAsOperator(SparseArrays.blockdiag(_sparse_of(b1), map(_sparse_of, bs)...), ts...)
end

function MatrixShapedOperators.blockdiag_operator(blocks::AbstractVector{<:_SparseLike})
    isempty(blocks) && throw(ArgumentError("blockdiag_operator requires at least one block"))
    ts = mapreduce(traitsof, traitset_blockdiag, blocks)
    return MatrixAsOperator(reduce(SparseArrays.blockdiag, map(_sparse_of, blocks)), ts...)
end


# Explicit host-side sparse materialization:

SparseArrays.sparse(op::MatrixShapedOperator) = SparseArrays.sparse(AbstractMatrix(op))

SparseArrays.sparse(op::MatrixAsOperator{<:Number,<:Tuple,<:AbstractMatrix}) = SparseArrays.sparse(parent(op))

SparseArrays.sparse(op::UniformScalingOperator) = SparseArrays.sparse(asmatrix(op))

SparseArrays.sparse(op::IdentityOperator) = SparseArrays.sparse(asmatrix(op))

SparseArrays.sparse(op::ZeroOperator) = SparseArrays.spzeros(Bool, op.m, op.n)

SparseArrays.sparse(op::BlockDiagOperator) =
    reduce(SparseArrays.blockdiag, map(SparseArrays.sparse, op.blocks))


end # module MatrixShapedOperatorsSparseArraysExt
