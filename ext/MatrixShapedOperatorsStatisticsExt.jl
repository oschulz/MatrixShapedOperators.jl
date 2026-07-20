# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).

module MatrixShapedOperatorsStatisticsExt

using Statistics

using MatrixShapedOperators: MatrixShapedOperator, MatrixShapedSum


# The generic mean would fold pairwise sums and grow tuple types with
# the number of terms, scale a single sum instead; scalar scaling
# preserves symmetry and value-refined definiteness:
function _mean_operator(ops)
    s = MatrixShapedSum(ops)
    # Follow the usual mean promotion; a real scalar also preserves
    # hermitianity for complex operators, and `eltype(s)(length(ops))`
    # would fail for `Bool` (identity operators):
    T = float(real(eltype(s)))
    return inv(T(length(ops))) * s
end

Statistics.mean(ops::AbstractVector{<:MatrixShapedOperator}) = _mean_operator(ops)
Statistics.mean(ops::Tuple{MatrixShapedOperator,Vararg{MatrixShapedOperator}}) = _mean_operator(ops)


end # module MatrixShapedOperatorsStatisticsExt
