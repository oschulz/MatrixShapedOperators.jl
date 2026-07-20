# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).

using MatrixShapedOperators
using Test
using TimerOutputs: @timed_testset

using LinearAlgebra
using SparseArrays

using MatrixShapedOperators: Hermitianity, IsHermitian, PosDefNess, IsPosDef,
    IsPosSemiDefOnly, Unitarity, IsUnitary, RowRankNess, IsFullRowRank,
    Triangularity, IsDiagonal, BatchedMulStyle, BatchedMul

@timed_testset "structured_operators" begin
    d1 = randn(Float32, 4)
    d2 = randn(Float32, 3)
    x4 = randn(Float32, 4)
    X4 = randn(Float32, 4, 3)

    @timed_testset "diagonal_operator" begin
        dop = @inferred(diagonal_operator(d1))
        @test dop isa MatrixAsOperator{Float32,<:Tuple,<:Diagonal}
        @test parent(dop) == Diagonal(d1)
        @test size(dop) == (4, 4)
        @test dop * x4 ≈ Diagonal(d1) * x4
        @test dop * X4 ≈ Diagonal(d1) * X4
        @test dop' * x4 ≈ Diagonal(d1) * x4
        @test issymmetric(dop) && ishermitian(dop)
        @test Triangularity(dop) === IsDiagonal()
    end

    @timed_testset "blockdiag_operator" begin
        # all-diagonal blocks collapse into a single diagonal operator:
        bd = blockdiag_operator(diagonal_operator(d1), diagonal_operator(d2))
        @test bd isa MatrixAsOperator{Float32,<:Tuple,<:Diagonal}
        @test Matrix(bd) ≈ Matrix(Diagonal(vcat(d1, d2)))

        # plain Diagonal matrices and uniform scalings collapse as well:
        bdm = blockdiag_operator(Diagonal(d1), Diagonal(d2))
        @test bdm isa MatrixAsOperator{Float32,<:Tuple,<:Diagonal}
        @test parent(bdm) == Diagonal(vcat(d1, d2))
        bdx = blockdiag_operator(Diagonal(d1), diagonal_operator(d2), 2.0f0 * 𝟙(3))
        @test bdx isa MatrixAsOperator{Float32,<:Tuple,<:Diagonal}
        @test parent(bdx) == Diagonal(vcat(d1, d2, fill(2.0f0, 3)))
        # scaled diagonal-likes collapse as well:
        bsc = blockdiag_operator(2.0f0 * diagonal_operator(d1), 𝟙(2))
        @test bsc isa MatrixAsOperator{Float32,<:Tuple,<:Diagonal}
        @test parent(bsc) == Diagonal(vcat(2.0f0 .* d1, ones(Float32, 2)))
        @test blockdiag_operator(Diagonal(d1)) isa MatrixAsOperator{Float32,<:Tuple,<:Diagonal}
        bdv = blockdiag_operator([Diagonal(d1), Diagonal(d2)])
        @test bdv isa MatrixAsOperator{Float32,<:Tuple,<:Diagonal}
        @test parent(bdv) == Diagonal(vcat(d1, d2))

        # generic blocks, mixed operators and matrices, non-square:
        A = randn(Float32, 2, 4)
        B = randn(Float32, 3, 3)
        bd2 = blockdiag_operator(asoperator(A), B, diagonal_operator(d2))
        ref = zeros(Float32, 2 + 3 + 3, 4 + 3 + 3)
        ref[1:2, 1:4] = A
        ref[3:5, 5:7] = B
        ref[6:8, 8:10] = Matrix(Diagonal(d2))
        @test bd2 isa BlockDiagOperator{Float32}
        @test size(bd2) == size(ref)
        @test BatchedMulStyle(bd2) === BatchedMul()
        z = randn(Float32, size(ref, 2))
        Z = randn(Float32, size(ref, 2), 3)
        @test bd2 * z ≈ ref * z
        @test bd2 * Z ≈ ref * Z
        zl = randn(Float32, size(ref, 1))
        @test bd2' * zl ≈ ref' * zl
        @test Matrix(bd2) ≈ ref
        @test Matrix(bd2') ≈ ref'

        # row-Gram blocks combine into a row-Gram of block-diagonal factors:
        g1 = RowGramOperator(randn(Float32, 3, 5))
        g2 = RowGramOperator(diagonal_operator(d2))
        gbd = blockdiag_operator(g1, g2)
        @test gbd isa RowGramOperator
        @test Matrix(gbd) ≈ [Matrix(g1) zeros(Float32, 3, 3); zeros(Float32, 3, 3) Matrix(g2)]

        # single-block cases:
        dop = diagonal_operator(d1)
        @test blockdiag_operator(dop) === dop
        @test blockdiag_operator(g1) === g1
        @test Matrix(blockdiag_operator(B)) ≈ B

        # empty Gram-block vectors are rejected:
        @test_throws ArgumentError blockdiag_operator(RowGramOperator[])
        @test_throws ArgumentError blockdiag_operator(ColGramOperator[])

        @timed_testset "trait derivation" begin
            Σ1 = asoperator(randn(Float32, 3, 3), IsPosDef())
            Σ2 = asoperator(randn(Float32, 2, 2), IsPosSemiDefOnly())
            bpsd = blockdiag_operator(Σ1, Σ2)
            @test PosDefNess(bpsd) === IsPosSemiDefOnly()
            @test Hermitianity(bpsd) === IsHermitian()
            bpd = blockdiag_operator(Σ1, Σ1)
            @test PosDefNess(bpd) === IsPosDef()
            @test isposdef(bpd)

            q1 = asoperator(randn(Float32, 3, 3), IsUnitary())
            q2 = asoperator(randn(Float32, 2, 2), IsUnitary())
            @test Unitarity(blockdiag_operator(q1, q2)) === IsUnitary()
            @test RowRankNess(blockdiag_operator(q1, q2)) === IsFullRowRank()
        end

        @timed_testset "sparse blocks and sparse conversion" begin
            S1 = sprand(4, 5, 0.4)
            S2 = sprand(3, 3, 0.5)
            ref = Matrix(blockdiag(S1, S2))

            # sparse blocks collapse into a single wrapped sparse matrix:
            for bs in [
                blockdiag_operator(S1, S2),
                blockdiag_operator(asoperator(S1), asoperator(S2)),
                blockdiag_operator(S1, asoperator(S2)),
                blockdiag_operator([S1, S2]),
            ]
                @test bs isa MatrixAsOperator{Float64,<:Tuple,<:SparseMatrixCSC}
                @test Matrix(bs) ≈ ref
            end
            @test blockdiag_operator(asoperator(S2)) isa MatrixAsOperator{Float64,<:Tuple,<:SparseMatrixCSC}

            # declared traits combine via the blockdiag trait algebra:
            P1 = sparse(Symmetric(sprand(3, 3, 0.5) + 2.0 * I))
            bp = blockdiag_operator(asoperator(P1, IsPosDef()), asoperator(P1, IsPosDef()))
            @test PosDefNess(bp) === IsPosDef()

            # dense blocks stay lazy:
            @test blockdiag_operator(randn(2, 2), randn(3, 3)) isa BlockDiagOperator

            # explicit sparse materialization:
            @test sparse(asoperator(Matrix(S2))) == sparse(Matrix(S2))
            @test sparse(2.0 * 𝟙(3)) == sparse(2.0 * I(3))
            bd_lazy = blockdiag_operator(asoperator(randn(2, 3)), diagonal_operator(randn(3)))
            sp = sparse(bd_lazy)
            @test sp isa SparseMatrixCSC
            @test sp ≈ Matrix(bd_lazy)
            mf = MulFuncOperator{Float64}(x -> 2 .* x, x -> 2 .* x, (3, 3))
            @test sparse(mf) ≈ 2.0 * I(3)

            @test_throws ArgumentError blockdiag_operator(SparseMatrixCSC{Float64,Int}[])
        end

        @timed_testset "logabsdet and lower_cholesky" begin
            D1 = Diagonal([4.0, 9.0])
            L1 = LowerTriangular(randn(3, 3) + 5 * I)
            bd = blockdiag_operator(asoperator(D1), asoperator(Matrix(L1)))
            @test bd == BlockDiagOperator((asoperator(D1), asoperator(Matrix(L1))))
            @test occursin("blockdiag_operator", sprint(show, bd))
            @test hash(bd) == hash(BlockDiagOperator((asoperator(D1), asoperator(Matrix(L1)))))
            @test bd ≈ BlockDiagOperator((asoperator(copy(D1)), asoperator(Matrix(L1))))
            lad = logabsdet(bd)
            lad_ref = logabsdet(Matrix(bd))
            @test lad[1] ≈ lad_ref[1] && lad[2] ≈ lad_ref[2]
            @test_throws DimensionMismatch logabsdet(blockdiag_operator(asoperator(randn(2, 3)), asoperator(randn(3, 2))))

            # uniform scalings are diagonal-like and collapse as well:
            bdu = blockdiag_operator(asoperator(D1), 4.0 * 𝟙(2))
            @test bdu isa MatrixAsOperator{Float64,<:Tuple,<:Diagonal}
            @test parent(bdu) == Diagonal([4.0, 9.0, 4.0, 4.0])

            # square triangular blocks make the block-diagonal
            # structurally triangular, so triangular Gram factors keep
            # their Cholesky path through the blockdiag collapse:
            g1 = rowgram_operator(LowerTriangular([2.0 0.0; 1.0 3.0]))
            g2 = rowgram_operator(LowerTriangular([4.0 0.0; 0.5 1.0]))
            gbd = blockdiag_operator(g1, g2)
            Lb = lower_cholesky(gbd)
            @test Matrix(Lb) * Matrix(Lb)' ≈ Matrix(gbd)
            # rectangular blocks stay shape-agnostic:
            bdrect = BlockDiagOperator((asoperator(LowerTriangular([2.0 0.0; 1.0 3.0])), asoperator(randn(1, 2))))
            @test MatrixShapedOperators.Triangularity(bdrect) === MatrixShapedOperators.UnknownTriangularity()

            B2 = randn(3, 3)
            Σ2 = Matrix(Symmetric(B2 * B2' + I))
            bdc = blockdiag_operator(asoperator(D1), asoperator(Σ2, IsPosDef()))
            lc = lower_cholesky(bdc)
            @test lc isa BlockDiagOperator
            @test Matrix(lc) * Matrix(lc)' ≈ Matrix(bdc)
        end

        @timed_testset "inverse application" begin
            L3 = LowerTriangular(randn(3, 3) + 3 * I)
            bds = BlockDiagOperator((asoperator(L3), asoperator(Diagonal([2.0, 4.0]))))
            y5 = randn(5)
            Y5 = randn(5, 2)
            @test bds \ y5 ≈ Matrix(bds) \ y5
            @test bds \ Y5 ≈ Matrix(bds) \ Y5
            bdr = BlockDiagOperator((asoperator(randn(2, 3)), asoperator(randn(3, 2))))
            @test_throws DimensionMismatch bdr \ randn(5)
        end

        @timed_testset "complex element types" begin
            Cz = randn(ComplexF64, 3, 3)
            bdz = BlockDiagOperator((asoperator(Cz), asoperator(randn(ComplexF64, 2, 2))))
            @test Matrix(bdz') ≈ Matrix(bdz)'
            @test Matrix(transpose(bdz)) ≈ transpose(Matrix(bdz))
        end

        @timed_testset "structural verbs on vector storage" begin
            Lvs = [asoperator(LowerTriangular(randn(3, 3) + 3 * I)) for _ in 1:4]
            bdv2 = BlockDiagOperator(Lvs)
            yv = randn(12)
            @test bdv2 \ yv ≈ Matrix(bdv2) \ yv
            ladv = logabsdet(bdv2)
            ladv_ref = logabsdet(Matrix(bdv2))
            @test ladv[1] ≈ ladv_ref[1] && ladv[2] ≈ ladv_ref[2]

            dvs = [asoperator(Diagonal(rand(3) .+ 1)) for _ in 1:3]
            bdd = BlockDiagOperator(dvs)
            lcd = lower_cholesky(bdd)
            @test Matrix(lcd) * Matrix(lcd)' ≈ Matrix(bdd)
            rgf = rowgram_factor(bdd)
            @test Matrix(rgf * rgf') ≈ Matrix(bdd)
        end

        @timed_testset "vector blocks" begin
            ds = [randn(Float32, 3) for _ in 1:30]
            bdv = blockdiag_operator(map(diagonal_operator, ds))
            @test bdv isa MatrixAsOperator{Float32,<:Tuple,<:Diagonal}
            @test Matrix(bdv) ≈ Matrix(Diagonal(reduce(vcat, ds)))

            Bs = [randn(Float32, 2, 3) for _ in 1:10]
            bdg = blockdiag_operator(map(asoperator, Bs))
            @test size(bdg) == (20, 30)
            zz = randn(Float32, 30)
            ref2 = cat(Bs...; dims = (1, 2))
            @test bdg * zz ≈ ref2 * zz
            @test (bdg * randn(Float32, 30, 4)) isa AbstractMatrix{Float32}
            @test Matrix(bdg') ≈ ref2'

            gs = [RowGramOperator(randn(Float32, 3, 4)) for _ in 1:5]
            gbdv = blockdiag_operator(gs)
            @test gbdv isa RowGramOperator
            @test Matrix(gbdv) ≈ cat(map(Matrix, gs)...; dims = (1, 2))
            @test blockdiag_operator(gs[1:1]) === gs[1]

            @test blockdiag_operator([asoperator(B)]) isa MatrixShapedOperator
            @test_throws ArgumentError blockdiag_operator(MatrixShapedOperator{Float32}[])
            @test_throws ArgumentError blockdiag_operator(typeof(diagonal_operator(d1))[])

            # tuple vs. vector block storage is a performance choice
            # and does not affect equality or hashing, like for sums
            # and products:
            bdt = blockdiag_operator(asoperator(Bs[1]), asoperator(Bs[2]))
            bdv2 = blockdiag_operator(map(asoperator, Bs[1:2]))
            @test bdt == bdv2
            @test hash(bdt) == hash(bdv2)
            @test bdt ≈ bdv2
        end
    end
end
