# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).

"""
    MatrixShapedOperators

Linear operators with matrix shape and semantics, accessed through
application and operator algebra, not through element access:
operators combine lazily with operators (`*`, `+`) and eagerly with
arrays (`op * x`, column-wise `op * X`, solves via `op \\ x`).

MatrixShapedOperators is designed to be tracing- and GPU-friendly. Structural
and behavioral information (triangularity, symmetricity, positive-definiteness,
etc.) is captured at construction time and managed through type-stable traits.

Use [`asoperator`](@ref) to wrap matrices and other compatible objects as
operators.
"""
module MatrixShapedOperators

using LinearAlgebra

using Compat: @compat

include("matrix_shaped.jl")
include("operator_traits.jl")
include("cholesky_factor.jl")
include("matrix_as_operator.jl")
include("mulfunc_operator.jl")
include("scaled_operator.jl")
include("row_gram_operator.jl")
include("col_gram_operator.jl")
include("woodbury_operator.jl")
include("operator_arithmetic.jl")
include("identity_operator.jl")
include("zero_operator.jl")
include("structured_operators.jl")
include("test_operator.jl")
include("precompile.jl")

function __init__()
    Base.Experimental.register_error_hint(MethodError) do io, exc, _, _
        if exc.f === test_operator &&
           isnothing(Base.get_extension(MatrixShapedOperators, :MatrixShapedOperatorsTestExt))
            print(io, "\nDid you forget to load Test?")
        end
    end
end

end # module
