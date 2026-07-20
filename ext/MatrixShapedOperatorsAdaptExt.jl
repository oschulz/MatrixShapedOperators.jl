# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).

module MatrixShapedOperatorsAdaptExt

import Adapt

using MatrixShapedOperators: MatrixShapedOperators, MatrixAsOperator, MulFuncOperator,
    RowGramOperator, ColGramOperator, WoodburyOperator, WoodburyFactorization,
    MatrixShapedSum, MatrixShapedProduct,
    BlockDiagOperator, ScaledOperator, rowgram_factor, colgram_factor


# Structural adaption preserves the stored trait sets; component checks
# already happened at construction and are not repeated.

function Adapt.adapt_structure(to, op::MatrixAsOperator{T,TRS}) where {T,TRS}
    A = Adapt.adapt(to, parent(op))
    return MatrixAsOperator{eltype(A),TRS,typeof(A)}(A)
end

# The declared element type is trusted, not inspected, so it stays the
# contract of the adapted callables; callables that own numerical
# storage must preserve it under adaption:
function Adapt.adapt_structure(to, op::MulFuncOperator{T,TRS}) where {T,TRS}
    ovp = Adapt.adapt(to, op.ovp)
    vop = Adapt.adapt(to, op.vop)
    return MulFuncOperator{T,TRS,typeof(ovp),typeof(vop)}(ovp, vop, op.sz)
end

Adapt.adapt_structure(to, op::RowGramOperator) =
    RowGramOperator(Adapt.adapt(to, rowgram_factor(op)))

Adapt.adapt_structure(to, op::ColGramOperator) =
    ColGramOperator(Adapt.adapt(to, colgram_factor(op)))

function Adapt.adapt_structure(to, op::WoodburyOperator{T}) where T
    A = Adapt.adapt(to, op.A)
    B = Adapt.adapt(to, op.B)
    D = Adapt.adapt(to, op.D)
    U = promote_type(eltype(A), eltype(B), eltype(D))
    return WoodburyOperator{U,typeof(A),typeof(B),typeof(D)}(A, B, D)
end

Adapt.adapt_structure(to, op::MatrixShapedSum) =
    MatrixShapedSum(map(Base.Fix1(Adapt.adapt, to), op.terms))

Adapt.adapt_structure(to, op::MatrixShapedProduct) =
    MatrixShapedProduct(map(Base.Fix1(Adapt.adapt, to), op.factors))

Adapt.adapt_structure(to, op::BlockDiagOperator) =
    BlockDiagOperator(map(Base.Fix1(Adapt.adapt, to), op.blocks))

# Adapting the wrapped operator may change its storage precision; the
# scalar follows, so the scaling does not re-promote the compute type:
function Adapt.adapt_structure(to, op::ScaledOperator)
    inner = Adapt.adapt(to, op.op)
    s = eltype(inner) === eltype(op.op) ? op.s : _adapt_scalar(op.s, eltype(inner))
    return ScaledOperator(s, inner)
end

_adapt_scalar(s::Real, ::Type{T}) where {T<:Number} = convert(real(T), s)
_adapt_scalar(s::Complex, ::Type{T}) where {T<:Number} = convert(Complex{real(T)}, s)

Adapt.adapt_structure(to, fac::WoodburyFactorization) =
    WoodburyFactorization(Adapt.adapt(to, fac.F))

Adapt.adapt_structure(to, op::MatrixShapedOperators._ConjOperator) =
    MatrixShapedOperators._ConjOperator(Adapt.adapt(to, op.op))


end # module MatrixShapedOperatorsAdaptExt
