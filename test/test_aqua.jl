# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).

import Test
import Aqua
import MatrixShapedOperators

Test.@testset "Package ambiguities" begin
    Test.@test isempty(Test.detect_ambiguities(MatrixShapedOperators))
end # testset

Test.@testset "Aqua tests" begin
    Aqua.test_all(
        MatrixShapedOperators,
        ambiguities = true
    )
end # testset
