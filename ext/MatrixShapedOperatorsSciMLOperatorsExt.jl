# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).

module MatrixShapedOperatorsSciMLOperatorsExt

using SciMLOperators

import MatrixShapedOperators
using MatrixShapedOperators: MatrixShapedOperator, MulFuncOperator

using LinearAlgebra


# Adapts a plain multiplication function to the SciMLOperators
# operator application signatures. The in-place form is an
# interoperability shim, not a performance interface: it computes
# out-of-place and copies, but honors the 5-arg `mul!` contract that
# SciML solvers rely on:
struct _OpApplyFunc{F} <: Function
    f::F
end
(g::_OpApplyFunc)(v, u, p, t) = g.f(v)
(g::_OpApplyFunc)(w, v, u, p, t) = w .= g.f(v)

function _function_operator(::Type{T}, sz::Dims{2}, ovp, vop, sym::Bool, herm::Bool, posdef::Bool) where {T<:Number}
    # the in-place cache prototypes must hold computed results, so
    # neutral element types like Bool promote to their float type:
    Tp = float(T)
    return SciMLOperators.FunctionOperator(
        _OpApplyFunc(ovp), Vector{Tp}(undef, sz[2]), Vector{Tp}(undef, sz[1]);
        # a missing adjoint function stays missing, so `has_adjoint`
        # reports it honestly:
        op_adjoint = vop === nothing ? nothing : _OpApplyFunc(vop),
        islinear = true, isconstant = true, isinplace = true, outofplace = true,
        issymmetric = sym, ishermitian = herm, isposdef = posdef
    )
end

MatrixShapedOperators.check_mulfunc_operator_support(::Type{<:SciMLOperators.AbstractSciMLOperator}) = nothing

# The declarations are validated by the MulFuncOperator construction,
# the conversion applies the value-refined is* flags:
function MatrixShapedOperators.mulfunc_operator(
    ::Type{<:SciMLOperators.AbstractSciMLOperator}, ::Type{T}, sz::Dims{2}, ovp, vop, decls...
) where {T<:Number}
    return SciMLOperators.FunctionOperator(MulFuncOperator{T}(ovp, vop, sz, decls...))
end


# Operator conversions use the value-refined is* flags:

function SciMLOperators.FunctionOperator(op::MulFuncOperator{T}) where T
    # SciMLOperators has no representation for a forward-less operator:
    op.ovp === nothing && throw(ArgumentError(
        "conversion to a SciMLOperators operator requires a forward multiplication function, convert the adjoint operator instead"
    ))
    return _function_operator(
        T, size(op), op.ovp, op.vop, issymmetric(op), ishermitian(op), isposdef(op)
    )
end

function SciMLOperators.FunctionOperator(op::MatrixShapedOperator{T}) where T
    return _function_operator(
        T, size(op), Base.Fix1(*, op), Base.Fix1(*, adjoint(op)),
        issymmetric(op), ishermitian(op), isposdef(op)
    )
end

Base.convert(::Type{SciMLOperators.AbstractSciMLOperator}, op::MatrixShapedOperator) = SciMLOperators.FunctionOperator(op)


end # module MatrixShapedOperatorsSciMLOperatorsExt
