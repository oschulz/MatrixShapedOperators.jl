# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).

using MatrixShapedOperators
using Test
using TimerOutputs: @timed_testset

using LinearAlgebra
import InverseFunctions

@timed_testset "inverse_functions" begin
    A = randn(4, 6)

    # exact representation-level round trips on the operator domain:
    InverseFunctions.test_inverse(rowgram_operator, asoperator(A), compare = ==)
    InverseFunctions.test_inverse(colgram_operator, asoperator(A), compare = ==)
    InverseFunctions.test_inverse(rowgram_factor, rowgram_operator(A), compare = ==)
    InverseFunctions.test_inverse(colgram_factor, colgram_operator(A), compare = ==)

    # matrix arguments round-trip to the asoperator-normalized factor,
    # which compares equal to the matrix by value:
    InverseFunctions.test_inverse(rowgram_operator, A, compare = ==)
    InverseFunctions.test_inverse(colgram_operator, A, compare = ==)
    @test rowgram_factor(rowgram_operator(A)) === asoperator(A)
    @test colgram_factor(colgram_operator(A)) === asoperator(A)

    op = asoperator(LowerTriangular(randn(3, 3) + 4 * I))
    InverseFunctions.test_inverse(rowgram_operator, op, compare = ==)
end
