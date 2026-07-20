# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).

using MatrixShapedOperators
using LinearMaps
using Test
using TimerOutputs: @timed_testset

using LinearAlgebra

using MatrixShapedOperators: IsHermitian, IsPosDef

@timed_testset "linear_maps" begin
    A = randn(Float32, 4, 5)
    x_l = randn(Float32, 4)
    x_r = randn(Float32, 5)

    ovp = Base.Fix1(*, A)
    vop = Base.Fix1(*, A')
    sz = size(A)
    T = eltype(A)

    for OP in [LinearMap, FunctionMap]
        op = mulfunc_operator(OP, T, sz, ovp, vop)
        @test op isa FunctionMap{T}
        @test op * x_r == A * x_r
        @test op' * x_l == A' * x_l
        @test !issymmetric(op) && !ishermitian(op) && !isposdef(op)
    end

    S = A * A' + I
    sovp = Base.Fix1(*, S)
    sop = mulfunc_operator(FunctionMap, T, size(S), sovp, sovp, IsHermitian(), IsPosDef())
    @test issymmetric(sop) && ishermitian(sop) && isposdef(sop)

    @timed_testset "operator conversion" begin
        mfop = MulFuncOperator{T}(ovp, vop, sz)
        wop = asoperator(A)

        for lm in [
            LinearMap(mfop), LinearMap{T}(mfop), FunctionMap(mfop), FunctionMap{T}(mfop),
            convert(LinearMap, mfop), convert(LinearMap{T}, mfop),
            convert(FunctionMap, mfop), convert(FunctionMap{T}, mfop),
            LinearMap(wop), FunctionMap(wop)
        ]
            @test lm isa FunctionMap{T}
            @test lm * x_r ≈ A * x_r
            @test lm' * x_l ≈ A' * x_l
        end

        B = LinearMap(randn(Float32, 5, 4))
        @test Matrix(mfop * B) ≈ A * Matrix(B)
        C = LinearMap(randn(Float32, 3, 4))
        @test Matrix(C * mfop) ≈ Matrix(C) * A
    end

    @timed_testset "element type widening and declaration checks" begin
        lm = LinearMap{Float64}(𝟙(3))
        @test eltype(lm) == Float64
        x3 = randn(3)
        @test lm * x3 ≈ x3
        @test convert(LinearMap{Float64}, 2 * 𝟙(3)) * x3 ≈ 2 .* x3
        @test_throws ArgumentError LinearMap{Float32}(asoperator(randn(3, 3)))

        # the foreign seam validates square-only declarations like the
        # in-package seams:
        @test_throws ArgumentError mulfunc_operator(
            FunctionMap, Float64, (4, 3),
            x -> x[1:3], y -> vcat(y, 0.0), MatrixShapedOperators.IsHermitian()
        )

        # mixed sums and differences stay in the LinearMaps world like
        # mixed products:
        lmS = LinearMap(randn(4, 4))
        opS = asoperator(randn(4, 4))
        @test Matrix(lmS + opS) ≈ Matrix(lmS) + Matrix(opS)
        @test Matrix(opS + lmS) ≈ Matrix(opS) + Matrix(lmS)
        @test Matrix(lmS - opS) ≈ Matrix(lmS) - Matrix(opS)
        @test Matrix(opS - lmS) ≈ Matrix(opS) - Matrix(lmS)
    end

    @timed_testset "one-directional operators" begin
        # a missing adjoint direction converts to FunctionMap's native
        # missing-`fc` form; a missing forward direction has no
        # LinearMaps representation:
        fwd_only = LinearMap(mulfunc_operator(T, sz, ovp, nothing))
        @test fwd_only * x_r ≈ A * x_r
        @test_throws ErrorException fwd_only' * x_l
        @test_throws ArgumentError LinearMap(mulfunc_operator(T, sz, nothing, vop))
        @test_throws ArgumentError mulfunc_operator(FunctionMap, T, sz, nothing, vop)
    end
end
