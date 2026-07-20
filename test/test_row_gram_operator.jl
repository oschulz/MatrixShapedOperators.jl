# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).

using MatrixShapedOperators
using Test
using TimerOutputs: @timed_testset

using LinearAlgebra

using MatrixShapedOperators: Symmetricity, IsSymmetric, UnknownSymmetricity,
    Hermitianity, IsHermitian, PosDefNess, IsPosDef, IsPosSemiDefOnly,
    Triangularity, IsLowerTriangular, IsFullRowRank, _ispossemidef

@timed_testset "row_gram_operator" begin
    A = randn(Float32, 4, 6)
    G_ref = A * A'
    x = randn(Float32, 4)
    X = randn(Float32, 4, 3)

    # rowgram_operator is the preferred construction seam:
    @test @inferred(rowgram_operator(A)) == RowGramOperator(A)
    @test rowgram_operator(asoperator(A)') isa RowGramOperator
    @test Matrix(rowgram_operator(asoperator(A)')) ≈ A' * A

    @timed_testset "rowgram_factor fallbacks" begin
        # non-Gram psd operators answer rowgram_factor via lower_cholesky:
        dop = asoperator(Diagonal([4.0, 9.0]))
        @test parent(rowgram_factor(dop)) == Diagonal([2.0, 3.0])
        @test rowgram_factor(4.0 * 𝟙(3)) == 2.0 * 𝟙(3)

        B = randn(4, 4)
        Σ = Matrix(Symmetric(B * B' + I))
        Fd = rowgram_factor(asoperator(Σ, IsPosDef()))
        @test Matrix(Fd) * Matrix(Fd)' ≈ Σ

        bd = BlockDiagOperator((dop, asoperator(Σ)))
        Fb = rowgram_factor(bd)
        @test Fb isa BlockDiagOperator
        @test Matrix(Fb) * Matrix(Fb)' ≈ Matrix(bd)
    end

    for factor in [A, asoperator(A)]
        g = @inferred(RowGramOperator(factor))
        @test g isa RowGramOperator{Float32}
        @test rowgram_factor(g) === asoperator(factor)
        @test @inferred(size(g)) == (4, 4)
        @test g' === g
        @test @inferred(g * x) ≈ G_ref * x
        @test g * X ≈ G_ref * X
        @test Matrix(g) ≈ G_ref
    end

    # an anonymous lazy product of the factor applies like the Gram
    # operator itself, but without its structural traits - use
    # rowgram_operator to make the Gram structure explicit:
    B = rowgram_factor(RowGramOperator(A))
    @test (B * B') isa MatrixShapedProduct
    @test (B * B') * x ≈ G_ref * x

    g = RowGramOperator(A)
    @test Symmetricity(g) === IsSymmetric()
    @test Hermitianity(g) === IsHermitian()
    @test issymmetric(g) && ishermitian(g)
    # a dense factor gives no structural rank guarantee:
    @test PosDefNess(g) === IsPosSemiDefOnly()
    @test !isposdef(g)
    @test _ispossemidef(g)

    # a structurally full-row-rank factor makes the Gram posdef:
    gr = RowGramOperator(asoperator(A, IsFullRowRank()))
    @test PosDefNess(gr) === IsPosDef()
    @test isposdef(gr)
    gu = RowGramOperator(UnitLowerTriangular(randn(3, 3)))
    @test isposdef(gu)

    # value refinement: square triangular factor with non-zero diagonal:
    Lp = LowerTriangular(randn(3, 3) + 5 * I)
    @test isposdef(RowGramOperator(Lp))
    @test isposdef(RowGramOperator(asoperator(Lp)))
    Ls = LowerTriangular([1.0 0.0; 2.0 0.0])
    @test !isposdef(RowGramOperator(Ls))
    @test !isposdef(RowGramOperator(randn(3, 3)))

    @timed_testset "logabsdet" begin
        Lq = LowerTriangular(randn(4, 4) + 5 * I)
        gq = RowGramOperator(Lq)
        lad = logabsdet(gq)
        lad_ref = logabsdet(Matrix(Lq) * Matrix(Lq)')
        @test lad[1] ≈ lad_ref[1] && lad[2] == 1.0
        @test_throws ArgumentError logabsdet(RowGramOperator(randn(3, 5)))
    end

    # column-Gram via the adjoint factor:
    W = randn(Float32, 4, 6)
    gc = RowGramOperator(asoperator(W)')
    @test size(gc) == (6, 6)
    @test Matrix(gc) ≈ W' * W

    @test occursin("rowgram_operator", sprint(show, g))
    @test RowGramOperator(A) == RowGramOperator(A)
    @test RowGramOperator(A) != RowGramOperator(A .+ 1)
    @test hash(RowGramOperator(copy(A))) == hash(RowGramOperator(A))
    @test RowGramOperator(copy(A)) ≈ RowGramOperator(A)
    @test RowGramOperator(A) ≈ RowGramOperator(A .+ 1.0f-7) rtol = 1.0f-3

    @timed_testset "complex element types" begin
        C = randn(ComplexF64, 3, 5)
        gz = RowGramOperator(C)
        z = randn(ComplexF64, 3)
        @test gz * z ≈ (C * C') * z
        @test Symmetricity(gz) === UnknownSymmetricity()
        @test Hermitianity(gz) === IsHermitian()
        @test ishermitian(gz) && !issymmetric(gz)
    end

    @timed_testset "inverse application" begin
        Lr = LowerTriangular(randn(4, 4) + 5 * I)
        gr2 = RowGramOperator(Lr)
        yr = randn(4)
        @test gr2 \ yr ≈ Matrix(gr2) \ yr
        @test gr2 \ (gr2 * yr) ≈ yr
        @test_throws DimensionMismatch RowGramOperator(randn(3, 5)) \ randn(3)
    end

    @timed_testset "lower_cholesky" begin
        # a triangular factor with positive diagonal passes through
        # (factors are stored asoperator-normalized):
        L = LowerTriangular(randn(4, 4) + 5 * I)
        gl = RowGramOperator(L)
        @test lower_cholesky(gl) isa MatrixAsOperator
        @test asmatrix(lower_cholesky(gl)) === L

        # negative diagonal entries are canonicalized by column signs:
        Ln = LowerTriangular(Matrix(L) * Diagonal([1, -1, 1, -1]))
        gn = RowGramOperator(Ln)
        Lc = asmatrix(lower_cholesky(gn))
        @test all(>(0), diag(Lc))
        @test Lc * Lc' ≈ Matrix(gn)

        # complex factors are canonicalized by column phases:
        Lz = LowerTriangular(randn(ComplexF64, 3, 3) + 5 * I)
        gz = RowGramOperator(Lz)
        Lzc = asmatrix(lower_cholesky(gz))
        # real positive diagonal up to floating-point rounding:
        @test all(x -> abs(imag(x)) < sqrt(eps()) * abs(x) && real(x) > 0, diag(Lzc))
        @test Lzc * Lzc' ≈ Matrix(gz)

        # zero diagonal entries (singular psd factors) keep their column:
        Ls = LowerTriangular([1.0 0.0; 2.0 0.0])
        Lsc = asmatrix(lower_cholesky(RowGramOperator(Ls)))
        @test Lsc * Lsc' ≈ Ls * Ls'

        # no structural factor, no structural lower_cholesky:
        @test_throws ArgumentError lower_cholesky(RowGramOperator(randn(3, 3)))
    end

    @timed_testset "canonicalization and singular factors" begin
        # a singular square factor propagates the zero determinant sign:
        Ls = LowerTriangular([1.0 0 0; 2 0 0; 3 4 5])
        @test logabsdet(rowgram_operator(Ls)) == (-Inf, 0.0)

        # complex canonical factors have an exactly real non-negative
        # diagonal, and canonicalization is idempotent in value:
        Lz = LowerTriangular(ComplexF64[2im 0; 1 3-4im])
        g = rowgram_operator(asoperator(Lz))
        Lc = lower_cholesky(g)
        d = diag(Matrix(Lc))
        @test all(x -> isreal(x) && real(x) >= 0, d)
        @test Matrix(Lc) * Matrix(Lc)' ≈ Matrix(g)
        Lc2 = MatrixShapedOperators._canonical_chol(Lc)
        @test Matrix(Lc2) == Matrix(Lc)

        # canonicalization composes through zero and conjugated factors:
        @test lower_cholesky(rowgram_operator(𝟘(3))) === 𝟘(3)
        cz = conj(asoperator(Lz))
        Lcc = lower_cholesky(rowgram_operator(cz))
        @test Matrix(Lcc) * Matrix(Lcc)' ≈ Matrix(rowgram_operator(cz))

        # capability errors are ArgumentErrors, not MethodErrors:
        sq = asoperator(randn(3, 3))
        @test_throws ArgumentError rowgram_factor(sq + sq)
        @test_throws ArgumentError colgram_factor(sq + sq)
    end
end
