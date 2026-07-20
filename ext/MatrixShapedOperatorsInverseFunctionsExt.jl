# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).

module MatrixShapedOperatorsInverseFunctionsExt

import InverseFunctions

using MatrixShapedOperators: rowgram_operator, rowgram_factor,
    colgram_operator, colgram_factor


# Involutive pairs; the factor-to-operator direction is an exact
# inverse on Gram operators, which store their factor:

InverseFunctions.inverse(::typeof(rowgram_operator)) = rowgram_factor
InverseFunctions.inverse(::typeof(rowgram_factor)) = rowgram_operator
InverseFunctions.inverse(::typeof(colgram_operator)) = colgram_factor
InverseFunctions.inverse(::typeof(colgram_factor)) = colgram_operator


end # module MatrixShapedOperatorsInverseFunctionsExt
