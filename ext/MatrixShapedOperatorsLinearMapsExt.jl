# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).

module MatrixShapedOperatorsLinearMapsExt

using LinearMaps

import MatrixShapedOperators
using MatrixShapedOperators: MatrixShapedOperator, MulFuncOperator

using LinearAlgebra


# The declarations are validated by the MulFuncOperator construction,
# the conversion applies the value-refined is* flags:
function MatrixShapedOperators.mulfunc_operator(
    ::Type{<:Union{LinearMap,FunctionMap}}, ::Type{T}, sz::Dims{2}, ovp, vop, decls...
) where {T<:Number}
    return FunctionMap{T}(MulFuncOperator{T}(ovp, vop, sz, decls...))
end


function LinearMaps.FunctionMap{T}(op::MulFuncOperator{T}) where {T<:Number}
    # a missing adjoint function is FunctionMap's native missing-`fc`
    # form, but LinearMaps has no representation for a forward-less
    # operator:
    op.ovp === nothing && throw(ArgumentError(
        "conversion to a LinearMap requires a forward multiplication function, convert the adjoint operator instead"
    ))
    FunctionMap{T,false}(
        op.ovp, op.vop, size(op)...;
        issymmetric = issymmetric(op), ishermitian = ishermitian(op), isposdef = isposdef(op)
    )
end

# A widening element type is accepted, the operator application
# promotes to it:
function LinearMaps.FunctionMap{T}(op::MatrixShapedOperator) where {T<:Number}
    promote_type(T, eltype(op)) === T || throw(ArgumentError(
        "can't convert operator with element type $(eltype(op)) to a LinearMap with narrower element type $T"
    ))
    FunctionMap{T,false}(
        Base.Fix1(*, op), Base.Fix1(*, adjoint(op)), size(op)...;
        issymmetric = issymmetric(op), ishermitian = ishermitian(op), isposdef = isposdef(op)
    )
end

LinearMaps.FunctionMap(op::MatrixShapedOperator{T}) where T = LinearMaps.FunctionMap{T}(op)

LinearMaps.LinearMap{T}(op::MatrixShapedOperator) where {T<:Number} = LinearMaps.FunctionMap{T}(op)
LinearMaps.LinearMap(op::MatrixShapedOperator{T}) where T = LinearMaps.LinearMap{T}(op)

Base.convert(::Type{LinearMaps.FunctionMap{T}}, op::MatrixShapedOperator) where {T<:Number} = LinearMaps.FunctionMap{T}(op)
Base.convert(::Type{LinearMaps.FunctionMap}, op::MatrixShapedOperator) = LinearMaps.FunctionMap(op)
Base.convert(::Type{LinearMaps.LinearMap{T}}, op::MatrixShapedOperator) where {T<:Number} = LinearMaps.LinearMap{T}(op)
Base.convert(::Type{LinearMaps.LinearMap}, op::MatrixShapedOperator) = LinearMaps.LinearMap(op)

# Mixed products, sums and differences stay in the LinearMaps world:
Base.:(*)(A::LinearMaps.LinearMap{<:Number}, B::MatrixShapedOperator) = A * LinearMaps.LinearMap(B)
Base.:(*)(A::MatrixShapedOperator, B::LinearMaps.LinearMap{<:Number}) = LinearMaps.LinearMap(A) * B
Base.:(+)(A::LinearMaps.LinearMap{<:Number}, B::MatrixShapedOperator) = A + LinearMaps.LinearMap(B)
Base.:(+)(A::MatrixShapedOperator, B::LinearMaps.LinearMap{<:Number}) = LinearMaps.LinearMap(A) + B
Base.:(-)(A::LinearMaps.LinearMap{<:Number}, B::MatrixShapedOperator) = A - LinearMaps.LinearMap(B)
Base.:(-)(A::MatrixShapedOperator, B::LinearMaps.LinearMap{<:Number}) = LinearMaps.LinearMap(A) - B


end # module MatrixShapedOperatorsLinearMapsExt
