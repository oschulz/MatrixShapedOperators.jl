# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).

using MatrixShapedOperators
using Test
using TimerOutputs: @timed_testset

using LinearAlgebra

using MatrixShapedOperators: Symmetricity, IsSymmetric, UnknownSymmetricity,
    Hermitianity, IsHermitian, UnknownHermitianity,
    PosDefNess, IsPosDef, UnknownPosDefNess,
    Triangularity, RowRankNess, IsFullRowRank, UnknownRowRankNess,
    BatchedMulStyle, BatchedMul, ColumnwiseMul, traitsof

# A multiplication function with native batched application:
struct _DiagMulTestFunc{T<:AbstractVector} <: Function
    d::T
end
(m::_DiagMulTestFunc)(x) = m.d .* x
MatrixShapedOperators.BatchedMulStyle(::_DiagMulTestFunc) = BatchedMul()

@timed_testset "mulfunc_operator" begin
    A = randn(Float32, 4, 5)
    ovp = Base.Fix1(*, A)
    vop = Base.Fix1(*, A')
    x_l = randn(Float32, 4)
    x_r = randn(Float32, 5)
    X_l = randn(Float32, 4, 3)
    X_r = randn(Float32, 5, 3)

    op = @inferred(MulFuncOperator{Float32}(ovp, vop, size(A)))
    @test op isa MulFuncOperator{Float32,Tuple{}}
    @test op == MulFuncOperator{Float32}(ovp, vop, (4, 5))
    @test @inferred(mulfunc_operator(Float32, size(A), ovp, vop)) == op
    @test @inferred(mulfunc_operator(MulFuncOperator, Float32, size(A), ovp, vop)) == op
    @test mulfunc_operator(Matrix, Float32, size(A), ovp, vop) ≈ A

    @test @inferred(eltype(op)) == Float32
    @test @inferred(size(op)) == (4, 5)
    @test traitsof(op) === ()
    @test !issymmetric(op) && !ishermitian(op) && !isposdef(op)
    @test occursin("mulfunc_operator", sprint(show, op))

    @test @inferred(op * x_r) == A * x_r
    @test @inferred(op * x_r) == A * x_r
    @test op * X_r ≈ A * X_r
    @test op' * x_l == A' * x_l
    @test op' * X_l ≈ A' * X_l
    @test op'' == op
    @test op'.ovp === vop && op'.vop === ovp

    @test BatchedMulStyle(op) === ColumnwiseMul()

    @timed_testset "trait declarations" begin
        S = A * A' + I
        sovp = Base.Fix1(*, S)
        sop = MulFuncOperator{Float32}(sovp, sovp, size(S), IsHermitian(), IsPosDef())
        @test Hermitianity(sop) === IsHermitian()
        @test PosDefNess(sop) === IsPosDef()
        @test RowRankNess(sop) === IsFullRowRank()
        @test Symmetricity(sop) === UnknownSymmetricity()
        # real hermitian implies symmetric at the is* level:
        @test ishermitian(sop) && isposdef(sop) && issymmetric(sop)
        @test Hermitianity(sop') === IsHermitian()
        @test PosDefNess(sop') === IsPosDef()

        # square-only structure declarations are rejected on non-square sizes:
        @test_throws ArgumentError MulFuncOperator{Float32}(ovp, vop, size(A), IsHermitian())
        # negative sizes are rejected like for the identity and zero operators:
        @test_throws ArgumentError MulFuncOperator{Float32}(ovp, vop, (-1, 5))
        @test_throws ArgumentError mulfunc_operator(Float32, (4, -1), ovp, vop)
        # rank declarations are fine on non-square sizes:
        rop = MulFuncOperator{Float32}(ovp, vop, size(A), IsFullRowRank())
        @test RowRankNess(rop) === IsFullRowRank()
        @test RowRankNess(rop') === UnknownRowRankNess()
    end

    @timed_testset "scalar scaling" begin
        S = A * A' + I
        sovp = Base.Fix1(*, S)
        sop = MulFuncOperator{Float32}(sovp, sovp, size(S), IsHermitian(), IsPosDef())

        sc = @inferred(2 * sop)
        @test sc isa ScaledOperator{Float32}
        @test sc * x_l ≈ 2 * (S * x_l)
        @test sc * (X_l .+ 0) ≈ 2 * (S * X_l)
        @test sc' * x_l ≈ 2 * (S * x_l)
        @test sop * 2 == sc
        @test (2.0 * sop) isa ScaledOperator{Float64}

        # scaling by an unknown value drops definiteness, keeps hermitianity:
        @test Hermitianity(sc) === IsHermitian()
        @test PosDefNess(sc) === UnknownPosDefNess()
        # a complex scalar also drops hermitianity:
        cc = (2 + 1im) * sop
        @test cc isa ScaledOperator{ComplexF32}
        @test Hermitianity(cc) === UnknownHermitianity()
        @test cc * complex.(x_l) ≈ (2 + 1im) * (S * x_l)
    end

    @timed_testset "batched mul style" begin
        d = randn(Float32, 5)
        dmul = _DiagMulTestFunc(d)
        dop = MulFuncOperator{Float32}(dmul, dmul, (5, 5))
        @test BatchedMulStyle(dop) === BatchedMul()
        @test dop * x_r ≈ d .* x_r
        @test dop * X_r ≈ Diagonal(d) * X_r
        @test dop' * X_r ≈ Diagonal(d)' * X_r
        @test BatchedMulStyle(2 * dop) === BatchedMul()
        @test (2 * dop) * X_r ≈ 2 * (Diagonal(d) * X_r)
    end

    @timed_testset "equality and requested types" begin
        f = x -> 2 .* x
        m1 = mulfunc_operator(Float64, (3, 3), f, f)
        # trait declarations do not participate in equality or hashing:
        @test m1 == mulfunc_operator(Float64, (3, 3), f, f, MatrixShapedOperators.IsHermitian())
        @test hash(m1) == hash(mulfunc_operator(Float64, (3, 3), f, f, MatrixShapedOperators.IsHermitian()))
        @test isapprox(m1, m1)
        @test m1 != mulfunc_operator(Float32, (3, 3), f, f)
        # the Matrix target honors the requested element type:
        M32 = mulfunc_operator(Matrix, Float32, (3, 3), f, f)
        @test M32 isa Matrix{Float32}
        @test M32 ≈ 2 * I(3)

        # a missing direction fails on application, not construction:
        mo = mulfunc_operator(Float64, (3, 3), f, nothing)
        x3 = randn(3)
        @test mo * x3 ≈ 2 .* x3
        @test_throws ArgumentError mo' * x3
        @test (mo')' * x3 ≈ 2 .* x3
        @test_throws ArgumentError mulfunc_operator(Float64, (3, 3), nothing, nothing)
    end
end
