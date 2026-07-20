# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).


abstract type MatrixShapedOperator{T<:Number} end

const MatrixShaped{T<:Number} = Union{AbstractMatrix{T},MatrixShapedOperator{T}}
