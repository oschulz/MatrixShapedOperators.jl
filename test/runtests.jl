# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).

import Test

using Test: @testset
using TimerOutputs: @timed_testset, print_timer

@timed_testset "Package MatrixShapedOperators" begin
    include("test_operator_traits.jl")
    include("test_matrix_shaped.jl")
    include("test_cholesky_factor.jl")
    include("test_matrix_as_operator.jl")
    include("test_mulfunc_operator.jl")
    include("test_row_gram_operator.jl")
    include("test_col_gram_operator.jl")
    include("test_woodbury_operator.jl")
    include("test_operator_arithmetic.jl")
    include("test_identity_operator.jl")
    include("test_zero_operator.jl")
    include("test_structured_operators.jl")
    include("test_test_operator.jl")
    include("test_linear_maps.jl")
    include("test_sciml_operators.jl")
    include("test_statistics.jl")
    include("test_chainrulescore.jl")
    include("test_inverse_functions.jl")
    include("test_adapt.jl")
    include("test_aqua.jl")
    # Reactant only supports 64-bit Linux and macOS (some of its
    # dependencies break already during precompilation elsewhere), so it
    # can't be a static test dependency. Tracing behavior is
    # platform-independent, so testing on Linux x64 with a current Julia
    # release suffices; Julia 1.10's parallel precompilation is prone to
    # deadlock on Reactant's large extension set:
    if Sys.islinux() && Sys.ARCH === :x86_64 &&
       isempty(VERSION.prerelease) && VERSION >= v"1.11"
        import Pkg
        Base.identify_package("Reactant") === nothing && Pkg.add("Reactant")
        include("test_reactant.jl")
    end
    include("test_docs.jl")
end # testset

print_timer()
println()
