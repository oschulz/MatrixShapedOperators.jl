# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).

module MatrixShapedOperatorsChainRulesCoreExt

using ChainRulesCore: @non_differentiable

import MatrixShapedOperators
using MatrixShapedOperators: MatrixShapedOperator,
    Symmetricity, Hermitianity, PosDefNess, Triangularity, Unitarity, RowRankNess,
    BatchedMulStyle

import LinearAlgebra


# Trait queries, the trait algebra and structural checks carry no
# derivative information:

@non_differentiable Symmetricity(::Any)
@non_differentiable Hermitianity(::Any)
@non_differentiable PosDefNess(::Any)
@non_differentiable Triangularity(::Any)
@non_differentiable Unitarity(::Any)
@non_differentiable RowRankNess(::Any)
@non_differentiable BatchedMulStyle(::Any)

@non_differentiable MatrixShapedOperators.NumberDomain(::Any)
@non_differentiable MatrixShapedOperators.traitsof(::Any)
@non_differentiable MatrixShapedOperators.traitset(::Any...)
@non_differentiable MatrixShapedOperators.gettrait(::Any, ::Any)
@non_differentiable MatrixShapedOperators.trait_category(::Any)
@non_differentiable MatrixShapedOperators.unknowntrait(::Any)

@non_differentiable MatrixShapedOperators.trait_add(::Any, ::Any)
@non_differentiable MatrixShapedOperators.trait_mul(::Any, ::Any)
@non_differentiable MatrixShapedOperators.trait_blockdiag(::Any, ::Any)
@non_differentiable MatrixShapedOperators.trait_adjoint(::Any)
@non_differentiable MatrixShapedOperators.trait_scale(::Any)
@non_differentiable MatrixShapedOperators.traitset_add(::Any, ::Any)
@non_differentiable MatrixShapedOperators.traitset_mul(::Any, ::Any)
@non_differentiable MatrixShapedOperators.traitset_blockdiag(::Any, ::Any)
@non_differentiable MatrixShapedOperators.traitset_adjoint(::Any)
@non_differentiable MatrixShapedOperators.traitset_scale(::Any)

@non_differentiable Base.size(::MatrixShapedOperator)
@non_differentiable Base.size(::MatrixShapedOperator, ::Any)
@non_differentiable LinearAlgebra.issymmetric(::MatrixShapedOperator)
@non_differentiable LinearAlgebra.ishermitian(::MatrixShapedOperator)
@non_differentiable LinearAlgebra.isposdef(::MatrixShapedOperator)

@non_differentiable Base.size(::MatrixShapedOperators.WoodburyFactorization)
@non_differentiable Base.size(::MatrixShapedOperators.WoodburyFactorization, ::Any)
@non_differentiable LinearAlgebra.issuccess(::MatrixShapedOperators.WoodburyFactorization)

for S in (:One, :Zero)
    @eval begin
        @non_differentiable LinearAlgebra.issymmetric(::MatrixShapedOperators.$S)
        @non_differentiable LinearAlgebra.ishermitian(::MatrixShapedOperators.$S)
        @non_differentiable LinearAlgebra.isposdef(::MatrixShapedOperators.$S)
    end
end


end # module MatrixShapedOperatorsChainRulesCoreExt
