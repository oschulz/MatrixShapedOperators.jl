# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).

import Test
import Aqua
import MatrixShapedOperators

using Test: @testset
using TimerOutputs: @timed_testset

# Runs after the extension test files, so the loaded extension method
# tables are guarded as well:
@timed_testset "Package ambiguities" begin
    ext_mods = filter(
        !isnothing,
        Any[Base.get_extension(MatrixShapedOperators, name) for name in (
            :MatrixShapedOperatorsAdaptExt, :MatrixShapedOperatorsChainRulesCoreExt,
            :MatrixShapedOperatorsInverseFunctionsExt, :MatrixShapedOperatorsLinearMapsExt,
            :MatrixShapedOperatorsSciMLOperatorsExt, :MatrixShapedOperatorsSparseArraysExt,
            :MatrixShapedOperatorsStatisticsExt, :MatrixShapedOperatorsTestExt
        )]
    )
    Test.@test isempty(Test.detect_ambiguities(MatrixShapedOperators, ext_mods...))
end # testset

@timed_testset "Aqua tests" begin
    Aqua.test_all(
        MatrixShapedOperators,
        ambiguities = true
    )
end # testset
