# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).

using MatrixShapedOperators
using Test
using TimerOutputs: @timed_testset

using LinearAlgebra

using MatrixShapedOperators: Symmetricity, IsSymmetric, UnknownSymmetricity,
    Hermitianity, IsHermitian, PosDefNess, IsPosDef, IsPosSemiDefOnly,
    IsFullRowRank, _ispossemidef

@timed_testset "col_gram_operator" begin
    A = randn(Float32, 6, 4)
    G_ref = A' * A
    x = randn(Float32, 4)
    X = randn(Float32, 4, 3)

    # colgram_operator is the preferred construction seam:
    @test @inferred(colgram_operator(A)) == ColGramOperator(A)

    for factor in [A, asoperator(A)]
        g = @inferred(ColGramOperator(factor))
        @test g isa ColGramOperator{Float32}
        @test colgram_factor(g) === asoperator(factor)
        @test @inferred(size(g)) == (4, 4)
        @test g' === g
        @test @inferred(g * x) ≈ G_ref * x
        @test g * X ≈ G_ref * X
        @test Matrix(g) ≈ G_ref
    end

    g = ColGramOperator(A)
    @test Symmetricity(g) === IsSymmetric()
    @test Hermitianity(g) === IsHermitian()
    @test issymmetric(g) && ishermitian(g)
    @test PosDefNess(g) === IsPosSemiDefOnly()
    @test !isposdef(g)
    @test _ispossemidef(g)
    @test occursin("colgram_operator", sprint(show, g))

    @timed_testset "factor duality" begin
        # rowgram_factor of a column Gram is the adjoint factor and
        # colgram_factor is generically adjoint ∘ rowgram_factor:
        @test rowgram_factor(g) === asoperator(A)'
        gr = rowgram_operator(randn(Float32, 4, 6))
        @test colgram_factor(gr) == rowgram_factor(gr)'
        wp = woodbury_operator(Diagonal(rand(3) .+ 1), randn(3, 2), Symmetric(Matrix(2.0 * I(2))))
        Gc = colgram_factor(wp)
        @test Matrix(Gc)' * Matrix(Gc) ≈ Matrix(wp)

        # generic fallback on other psd operator types:
        Gu = colgram_factor(4.0 * 𝟙(3))
        @test Matrix(Gu)' * Matrix(Gu) ≈ 4.0 * I(3)
        bdp = BlockDiagOperator((asoperator(Diagonal([4.0, 9.0])), 2.0 * 𝟙(2)))
        Gb = colgram_factor(bdp)
        @test Matrix(Gb)' * Matrix(Gb) ≈ Matrix(bdp)
    end

    @timed_testset "structural posdef" begin
        # full column rank of A = full row rank of A', via adjoint traits:
        gq = ColGramOperator(asoperator(randn(Float32, 4, 4), MatrixShapedOperators.IsUnitary()))
        @test PosDefNess(gq) === IsPosDef()
        @test isposdef(gq)
        # value refinement via square triangular factors:
        U = UpperTriangular(randn(3, 3) + 5 * I)
        @test isposdef(ColGramOperator(U))
        @test !isposdef(ColGramOperator(randn(3, 3)))
    end

    @timed_testset "logabsdet" begin
        U = UpperTriangular(randn(4, 4) + 5 * I)
        lad = logabsdet(ColGramOperator(U))
        lad_ref = logabsdet(Matrix(U)' * Matrix(U))
        @test lad[1] ≈ lad_ref[1] && lad[2] == 1.0
        @test_throws ArgumentError logabsdet(ColGramOperator(randn(5, 3)))
    end

    @timed_testset "lower_cholesky" begin
        # an upper triangular factor makes A' the lower Cholesky factor:
        U = UpperTriangular(randn(4, 4) + 5 * I)
        gu = ColGramOperator(U)
        Lc = lower_cholesky(gu)
        @test Lc isa MatrixAsOperator
        @test Matrix(Lc) * Matrix(Lc)' ≈ Matrix(gu)
        Un = UpperTriangular(Diagonal([1, -1, 1, -1]) * Matrix(U))
        Lcn = lower_cholesky(ColGramOperator(Un))
        @test all(>(0), diag(asmatrix(Lcn)))
        @test Matrix(Lcn) * Matrix(Lcn)' ≈ Matrix(ColGramOperator(Un))
        @test_throws ArgumentError lower_cholesky(ColGramOperator(randn(3, 3)))
    end

    @timed_testset "inverse application" begin
        Uc = UpperTriangular(randn(4, 4) + 5 * I)
        gc2 = ColGramOperator(Uc)
        yc = randn(4)
        @test gc2 \ yc ≈ Matrix(gc2) \ yc
        @test gc2 \ (gc2 * yc) ≈ yc
        @test_throws DimensionMismatch ColGramOperator(randn(5, 3)) \ randn(3)
    end

    @timed_testset "blockdiag collapse" begin
        g1 = colgram_operator(randn(Float32, 5, 3))
        g2 = colgram_operator(randn(Float32, 2, 4))
        bd = blockdiag_operator(g1, g2)
        @test bd isa ColGramOperator
        @test Matrix(bd) ≈ cat(Matrix(g1), Matrix(g2); dims = (1, 2))
        @test blockdiag_operator(g1) === g1

        gs = [colgram_operator(randn(Float32, 3, 4)) for _ in 1:4]
        bdv = blockdiag_operator(gs)
        @test bdv isa ColGramOperator
        @test Matrix(bdv) ≈ cat(map(Matrix, gs)...; dims = (1, 2))
        @test blockdiag_operator(gs[1:1]) === gs[1]
        @test_throws ArgumentError blockdiag_operator(typeof(g1)[])
    end

    @timed_testset "complex element types" begin
        C = randn(ComplexF64, 5, 3)
        gz = ColGramOperator(C)
        z = randn(ComplexF64, 3)
        @test gz * z ≈ (C' * C) * z
        @test Symmetricity(gz) === UnknownSymmetricity()
        @test ishermitian(gz) && !issymmetric(gz)
    end
end
