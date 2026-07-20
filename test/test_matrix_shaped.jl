# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).

using MatrixShapedOperators
using Test
using TimerOutputs: @timed_testset

using LinearAlgebra

import MatrixShapedOperators as MSOs

# Minimal MatrixShapedOperator subtype, to test the generic fallbacks:
struct _WrappedTestOp{T<:Number,M<:AbstractMatrix{T}} <: MatrixShapedOperator{T}
    A::M
end
Base.size(op::_WrappedTestOp) = size(op.A)
Base.adjoint(op::_WrappedTestOp) = _WrappedTestOp(copy(adjoint(op.A)))
MSOs.explicit_mul_impl(op::_WrappedTestOp, x::AbstractVector{<:Number}) = op.A * x

@timed_testset "matrix_shaped" begin
    A = randn(Float32, 4, 5)
    x_l = randn(Float32, 4)
    x_r = randn(Float32, 5)
    X_l = randn(Float32, 4, 3)
    X_r = randn(Float32, 5, 3)

    op = _WrappedTestOp(A)

    @timed_testset "shape and eltype" begin
        @test op isa MatrixShapedOperator{Float32}
        @test op isa MatrixShaped{Float32}
        @test A isa MatrixShaped{Float32}
        @test eltype(op) == Float32
        @test eltype(typeof(op)) == Float32
        @test size(op) == (4, 5)
        @test size(op, 1) == 4 && size(op, 2) == 5
        @test size(op, 3) == 1
        @test_throws ArgumentError size(op, 0)
        @test ndims(op) == 2
        @test ndims(typeof(op)) == 2
    end

    @timed_testset "application" begin
        @test @inferred(op * x_r) ≈ A * x_r
        @test @inferred(op * x_r) ≈ A * x_r
        @test op * X_r ≈ A * X_r
        @test op' * x_l ≈ A' * x_l
        @test op' * X_l ≈ A' * X_l
        @test @inferred(x_l' * op) ≈ x_l' * A
        @test @inferred(transpose(x_l) * op) ≈ transpose(x_l) * A
        @test transpose(op) * x_l ≈ transpose(A) * x_l

        @test_throws DimensionMismatch op * x_l
        @test_throws DimensionMismatch op * X_l
        @test_throws DimensionMismatch op' * x_r

        Y = op * zeros(Float32, 5, 0)
        @test size(Y) == (4, 0)
        @test eltype(Y) == Float32
        @test eltype(op * zeros(Float64, 5, 0)) == Float64
    end

    @timed_testset "complex operators" begin
        C = randn(ComplexF64, 3, 3)
        opc = _WrappedTestOp(C)
        z = randn(ComplexF64, 3)
        @test opc * z ≈ C * z
        @test opc' * z ≈ C' * z
        @test z' * opc ≈ z' * C

        # lazy conjugation and complex transpose:
        @test conj(op) === op
        cc = conj(opc)
        @test cc isa MatrixShapedOperator{ComplexF64}
        @test cc * z ≈ conj(C) * z
        @test cc' * z ≈ conj(C)' * z
        @test conj(cc) === opc
        @test transpose(opc) * z ≈ transpose(C) * z
        @test transpose(z) * opc ≈ transpose(z) * C

        # conjugation preserves the structure traits:
        h = asoperator(Diagonal(complex.(randn(3))), MSOs.IsHermitian())
        @test MSOs.Hermitianity(conj(h)) === MSOs.IsHermitian()
        @test MSOs.Triangularity(conj(h)) === MSOs.IsDiagonal()
    end

    @timed_testset "matrices are data, operators are algebra" begin
        # matrix operands of * are applied eagerly, batch-wise:
        B = randn(Float32, 5, 3)
        @test op * B isa Matrix{Float32}
        @test op * B ≈ A * B
        @test B' * op' isa AbstractMatrix{Float32}
        @test B' * op' ≈ B' * A'
        @test_throws DimensionMismatch op * randn(Float32, 4, 4)
        @test_throws DimensionMismatch randn(Float32, 3, 3) * op

        # matrices enter the lazy algebra explicitly via asoperator:
        p = op * asoperator(B)
        @test p isa MatrixShapedProduct{Float32}
        @test Matrix(p) ≈ A * B

        S = randn(Float32, 4, 4)
        sq = _WrappedTestOp(S)
        s = sq + asoperator(S)
        @test s isa MatrixShapedSum{Float32}
        @test Matrix(s) ≈ S + S
        @test Matrix(asoperator(S) + sq) ≈ S + S
        @test_throws DimensionMismatch op + _WrappedTestOp(S)

        # adding a raw matrix would materialize implicit operators:
        @test_throws ArgumentError op + A
        @test_throws ArgumentError A + op
        @test_throws ArgumentError sq - S
        @test_throws ArgumentError S - sq
    end

    @timed_testset "scalar scaling" begin
        for sop in [3 * op, op * 3, (3 * I) * op, op * (3 * I)]
            @test sop isa MatrixShapedOperator
            @test sop * x_r ≈ 3 * (A * x_r)
        end
        @test (3.0 * op) isa MatrixShapedOperator{Float64}
        @test (op * I) isa ScaledOperator
        @test (op * I) * x_r ≈ A * x_r

        # nested scalings collapse:
        n23 = 2 * (3 * op)
        @test n23 isa ScaledOperator
        @test n23.s == 6 && n23.op === op

        # structural operations derive from the wrapped operator:
        S4 = randn(4, 4) + 5 * I
        sca = 2.5 * asoperator(S4)
        y4 = randn(4)
        lad = logabsdet(sca)
        lad_ref = logabsdet(2.5 * S4)
        @test lad[1] ≈ lad_ref[1] && lad[2] ≈ lad_ref[2]
        @test sca \ y4 ≈ (2.5 * S4) \ y4
        @test occursin("2.5 * ", sprint(show, sca))

        # is* queries refine by the scalar value:
        Σs = asoperator(S4 * S4' + I, MSOs.IsPosDef())
        @test isposdef(2 * Σs)
        @test !isposdef(-2 * Σs)
        @test ishermitian(2 * Σs) && !ishermitian((1 + 1im) * Σs)

        # empty operators have determinant one, even for zero scalings:
        @test logabsdet(0.0 * asoperator(zeros(0, 0))) == (0.0, 1.0)

        # Cholesky and Gram factors scale by sqrt(s) for real
        # non-negative scalings:
        Ls = lower_cholesky(2.5 * Σs)
        @test Matrix(Ls * Ls') ≈ 2.5 * Matrix(Σs)
        @test upper_cholesky(2.5 * Σs) == Ls'
        Fs = rowgram_factor(2.5 * Σs)
        @test Matrix(Fs * Fs') ≈ 2.5 * Matrix(Σs)
        @test_throws ArgumentError lower_cholesky(-2.5 * Σs)
        @test_throws ArgumentError rowgram_factor((1 + 1im) * Σs)
        g6 = rowgram_operator(randn(4, 6))
        Fg = rowgram_factor(2.0 * g6)
        @test Matrix(Fg * Fg') ≈ 2.0 * Matrix(g6)

        # scaled triangular Gram factors canonicalize by |s|:
        Lt = asoperator(LowerTriangular([2.0 0.0; 0.5 1.0]))
        gt = rowgram_operator(-2.5 * Lt)
        Lc = lower_cholesky(gt)
        @test Matrix(Lc * Lc') ≈ Matrix(gt)
        @test istril(Matrix(Lc)) && all(>=(0), diag(Matrix(Lc)))
    end

    @timed_testset "negation, subtraction and division" begin
        S = randn(Float32, 4, 4)
        R = randn(Float32, 4, 4)
        a = _WrappedTestOp(S)
        b = _WrappedTestOp(R)
        x = randn(Float32, 4)

        @test (-a) * x ≈ -(S * x)
        @test (a - b) * x ≈ (S - R) * x
        @test (a - asoperator(R)) * x ≈ (S - R) * x
        @test (asoperator(S) - b) * x ≈ (S - R) * x
        @test (a - I) * x ≈ (S - I) * x
        @test (2 * I - a) * x ≈ (2 * I - S) * x
        @test (a / 2) * x ≈ (S / 2) * x
        @test_throws DimensionMismatch op - a

        Σ = asoperator(S * S' + I, MSOs.IsPosDef())
        @test MSOs.PosDefNess(-Σ) === MSOs.UnknownPosDefNess()
        @test MSOs.Hermitianity(-Σ) === MSOs.IsHermitian()
    end

    @timed_testset "mul!" begin
        y = similar(x_l)
        @test mul!(y, op, x_r) === y
        @test y ≈ A * x_r
        Y = similar(X_l)
        @test mul!(Y, op, X_r) === Y
        @test Y ≈ A * X_r

        y2 = rand(Float32, 4)
        y2_ref = 2 * (A * x_r) + 3 * y2
        @test mul!(y2, op, x_r, 2, 3) === y2
        @test y2 ≈ y2_ref

        y3 = fill(Float32(NaN), 4)
        @test mul!(y3, op, x_r, 2, 0) ≈ 2 * (A * x_r)
    end

    @timed_testset "materialization" begin
        @test AbstractMatrix(op) ≈ A
        @test Matrix(op) ≈ A
        @test Matrix(op) isa Matrix{Float32}
        @test convert(AbstractMatrix, op) ≈ A
        @test convert(Matrix, op) ≈ A
    end

    @timed_testset "inverse application" begin
        S = randn(Float32, 4, 4)
        sq = _WrappedTestOp(S)
        y = randn(Float32, 4)

        # no structural inverse, Cholesky or determinant path for a
        # generic operator:
        @test_throws ArgumentError sq \ y
        @test_throws ArgumentError lower_cholesky(sq)
        @test_throws ArgumentError logabsdet(sq)
        # non-square operators and mismatched shapes are rejected:
        @test_throws DimensionMismatch op \ x_r
        @test_throws DimensionMismatch sq \ randn(Float32, 5)

        # unitary operators invert via their adjoint:
        u = mulfunc_operator(
            Float32, (4, 4), x -> circshift(x, 1), x -> circshift(x, -1), MSOs.IsUnitary()
        )
        @test u \ y ≈ circshift(y, -1)
        @test u \ (u * y) ≈ y

        # conjugated and scaled operators solve via their components:
        uz = mulfunc_operator(
            ComplexF64, (4, 4),
            x -> im .* circshift(x, 1), x -> -im .* circshift(x, -1), MSOs.IsUnitary()
        )
        cz = conj(uz)
        zc = randn(ComplexF64, 4)
        @test cz \ (cz * zc) ≈ zc
        @test (2 * u) \ y ≈ circshift(y, -1) ./ 2
    end

    @timed_testset "asoperator and asmatrix" begin
        @test asoperator(op) === op
        @test asoperator(A) isa MatrixAsOperator{Float32}
        @test asmatrix(A) === A
        @test asmatrix(asoperator(A)) === A
        # deliberately not defined for implicit operators:
        @test_throws MethodError asmatrix(op)
    end

    @timed_testset "funnel and protocol closure" begin
        S = randn(Float32, 4, 4)
        sq2 = asoperator(S)
        E = sq2 \ zeros(Float32, 4, 0)
        @test size(E) == (4, 0) && eltype(E) == Float32
        @test Matrix(sq2 / (2 * I)) ≈ S / 2
        @test Matrix((2 * I) \ sq2) ≈ S / 2
        Q = qr(randn(Float32, 4, 4)).Q
        @test Matrix(sq2 - Q) ≈ S - Matrix(asoperator(Q))
        @test Matrix(Q - sq2) ≈ Matrix(asoperator(Q)) - S
        Ar = randn(Float32, 3, 4)
        Y = zeros(Float32, 3, 4)
        @test mul!(Y, Ar, sq2) ≈ Ar * S
        Y2 = randn(Float32, 3, 4)
        Y2r = 2 .* (Ar * S) .+ 3 .* Y2
        @test mul!(Y2, Ar, sq2, 2, 3) ≈ Y2r
        # comparison is representational - different representations
        # compare unequal instead of erroring:
        @test !isapprox(sq2, 2 * sq2)
        @test !isapprox(sq2 + sq2, sq2 * sq2)
        @test MatrixShapedSum((sq2, sq2)) == MatrixShapedSum([sq2, sq2])
        @test hash(MatrixShapedSum((sq2, sq2))) == hash(MatrixShapedSum([sq2, sq2]))
        # non-square product capability errors are ArgumentErrors:
        P23 = asoperator(randn(2, 3)) * asoperator(randn(3, 2))
        @test_throws ArgumentError logabsdet(P23)
        @test_throws ArgumentError P23 \ randn(2)
    end
end
