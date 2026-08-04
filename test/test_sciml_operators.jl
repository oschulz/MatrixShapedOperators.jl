# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).

using MatrixShapedOperators
using SciMLOperators
using Test
using TimerOutputs: @timed_testset

using LinearAlgebra

using MatrixShapedOperators: IsHermitian, IsPosDef

@timed_testset "sciml_operators" begin
    @test MatrixShapedOperators.check_mulfunc_operator_support(SciMLOperators.AbstractSciMLOperator) === nothing
    @test MatrixShapedOperators.check_mulfunc_operator_support(SciMLOperators.FunctionOperator) === nothing

    A = randn(Float32, 4, 5)
    x_l = randn(Float32, 4)
    x_r = randn(Float32, 5)

    ovp = Base.Fix1(*, A)
    vop = Base.Fix1(*, A')
    sz = size(A)
    T = eltype(A)

    for OP in [SciMLOperators.AbstractSciMLOperator, SciMLOperators.FunctionOperator]
        op = mulfunc_operator(OP, T, sz, ovp, vop)
        @test op isa SciMLOperators.FunctionOperator
        @test eltype(op) == T
        @test size(op) == sz
        @test SciMLOperators.islinear(op)
        @test op * x_r ≈ A * x_r
        @test op' * x_l ≈ A' * x_l
        @test mul!(similar(x_l), op, x_r) ≈ A * x_r
    end

    S = A * A' + I
    sovp = Base.Fix1(*, S)
    sop = mulfunc_operator(SciMLOperators.FunctionOperator, T, size(S), sovp, sovp, IsHermitian(), IsPosDef())
    @test ishermitian(sop) && isposdef(sop)

    @timed_testset "operator conversion" begin
        mfop = MulFuncOperator{T}(ovp, vop, sz)
        wop = asoperator(A)

        for op in [
            SciMLOperators.FunctionOperator(mfop),
            SciMLOperators.FunctionOperator(wop),
            convert(SciMLOperators.AbstractSciMLOperator, mfop)
        ]
            @test op isa SciMLOperators.FunctionOperator
            @test op * x_r ≈ A * x_r
            @test op' * x_l ≈ A' * x_l
        end

        # a missing adjoint direction converts to SciML's native
        # missing-adjoint form instead of a broken adjoint function; a
        # missing forward direction has no SciML representation:
        fwd_only = SciMLOperators.FunctionOperator(mulfunc_operator(T, sz, ovp, nothing))
        @test fwd_only * x_r ≈ A * x_r
        @test !SciMLOperators.has_adjoint(fwd_only)
        @test_throws ArgumentError SciMLOperators.FunctionOperator(
            mulfunc_operator(T, sz, nothing, Base.Fix1(*, A'))
        )
        @test_throws ArgumentError mulfunc_operator(
            SciMLOperators.FunctionOperator, T, sz, nothing, Base.Fix1(*, A')
        )
    end

    @timed_testset "5-arg mul! and declaration checks" begin
        # alpha/beta are honored by converted operators of all kinds:
        A4 = randn(4, 4)
        Σ4 = A4 * A4' + I
        zoo = [
            asoperator(A4), 2.5 * asoperator(A4),
            asoperator(A4) + asoperator(Σ4), asoperator(A4) * asoperator(Σ4),
            rowgram_operator(randn(4, 6)),
            woodbury_operator(Diagonal(rand(4) .+ 1), randn(4, 2), Symmetric(Matrix(2.0I(2)))),
            blockdiag_operator(asoperator(randn(2, 2)), asoperator(randn(2, 2))),
            𝟙(4), 2.5 * 𝟙(4), ZeroOperator(4, 4)
        ]
        for mso in zoo
            so = convert(SciMLOperators.AbstractSciMLOperator, mso)
            x = randn(4)
            y = randn(4)
            yr = 2.0 .* (Matrix(mso) * x) .+ 3.0 .* y
            @test mul!(copy(y), so, x, 2.0, 3.0) ≈ yr
        end

        # the foreign seam validates square-only declarations like the
        # in-package seams:
        @test_throws ArgumentError mulfunc_operator(
            SciMLOperators.FunctionOperator, Float64, (4, 3),
            x -> x[1:3], y -> vcat(y, 0.0), MatrixShapedOperators.IsHermitian()
        )
    end
end
