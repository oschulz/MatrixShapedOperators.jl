# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).

using MatrixShapedOperators
using Test
using TimerOutputs: @timed_testset

using LinearAlgebra

using MatrixShapedOperators: Symmetricity, IsSymmetric, Hermitianity, IsHermitian,
    UnknownHermitianity, PosDefNess, IsPosDef, IsPosSemiDefOnly, UnknownPosDefNess,
    Triangularity, IsDiagonal, IsLowerTriangular, IsUpperTriangular,
    Unitarity, IsUnitary, RowRankNess, IsFullRowRank

@timed_testset "operator_arithmetic" begin
    @timed_testset "UniformScalingOperator" begin
        x = randn(Float32, 5)
        X = randn(Float32, 5, 3)

        u = 2.0f0 * 𝟙(5)
        @test u isa MatrixShapedOperator{Float32}
        @test u isa UniformScalingOperator{Float32}
        @test u === ScaledOperator(2.0f0, 𝟙(5))
        @test 2.0f0 * 𝟙(Int32(5)) == u
        @test size(u) == (5, 5)
        @test u * x ≈ 2 * x
        @test u * X ≈ 2 * X
        @test u' == u
        @test Symmetricity(u) === IsSymmetric()
        @test Hermitianity(u) === IsHermitian()
        @test Triangularity(u) === IsDiagonal()
        @test issymmetric(u) && ishermitian(u) && isposdef(u)
        @test !isposdef(-1.0f0 * 𝟙(5))
        @test !isposdef(0.0f0 * 𝟙(5))
        @test Matrix(u) ≈ 2 * I(5)
        @test asmatrix(u) == Diagonal(fill(2.0f0, 5))
        @test lower_cholesky(u) == sqrt(2.0f0) * 𝟙(5)
        @test occursin("𝟙(5)", sprint(show, u))

        @test u \ x ≈ x ./ 2
        @test u \ X ≈ X ./ 2

        @test hash(u) == hash(2.0f0 * 𝟙(5))
        @test u ≈ (2.0f0 + 1.0f-7) * 𝟙(5) rtol = 1.0f-3

        # scalar scaling stays a uniform scaling with working
        # structural operations:
        su = 3 * u
        @test su isa UniformScalingOperator
        @test su == 6.0f0 * 𝟙(5)
        @test logabsdet(2 * (3.0 * 𝟙(3)))[1] ≈ 3 * log(6.0)

        lad = logabsdet(u)
        @test lad[1] ≈ 5 * log(2.0f0) && lad[2] == 1
        ladn = logabsdet(-2.0 * 𝟙(3))
        @test ladn[1] ≈ 3 * log(2.0) && ladn[2] == -1.0

        uz = (2.0 + 1.0im) * 𝟙(3)
        @test uz' == (2.0 - 1.0im) * 𝟙(3)
        @test Symmetricity(uz) === IsSymmetric()
        @test Hermitianity(uz) === UnknownHermitianity()
        @test !isposdef(uz)

        ladz = logabsdet((1.0 + 1.0im) * 𝟙(2))
        @test ladz[1] ≈ 2 * log(abs(1.0 + 1.0im))
        @test ladz[2] ≈ sign(1.0 + 1.0im)^2

        # empty operators have determinant one, even for zero scalings:
        @test logabsdet(0.0 * 𝟙(0)) == (0.0, 1.0)

        @test_throws ArgumentError 2.0 * 𝟙(-1)

        # Gram factors of uniform scalings canonicalize by |λ|:
        @test lower_cholesky(rowgram_operator(-2.0 * 𝟙(3))) == 2.0 * 𝟙(3)
        @test lower_cholesky(rowgram_operator(3.0im * 𝟙(2))) == 3.0 * 𝟙(2)
    end

    @timed_testset "MatrixShapedSum" begin
        A = randn(Float32, 5, 5)
        B = randn(Float32, 5, 5)
        op_a = asoperator(A)
        op_b = asoperator(B)
        x = randn(Float32, 5)
        X = randn(Float32, 5, 3)

        s = op_a + op_b
        @test s isa MatrixShapedSum{Float32}
        @test size(s) == (5, 5)
        @test s * x ≈ (A + B) * x
        @test s * X ≈ (A + B) * X
        @test Matrix(s) ≈ A + B
        @test s == op_a + op_b
        @test hash(s) == hash(asoperator(copy(A)) + op_b)
        @test s ≈ asoperator(copy(A)) + op_b
        @test occursin("MatrixShapedSum", sprint(show, s))
        @test s' * x ≈ (A + B)' * x

        # no structural determinant or Cholesky factor for sums:
        @test_throws ArgumentError logabsdet(s)
        @test_throws ArgumentError lower_cholesky(s)

        # UniformScaling terms become UniformScalingOperator terms:
        si = op_a + op_b + I
        @test si isa MatrixShapedSum
        @test si.terms[3] isa UniformScalingOperator
        @test si * x ≈ (A + B + I) * x
        @test Matrix(I + op_a) ≈ A + I
        @test Matrix(op_a + 2 * I) ≈ A + 2 * I
        @test Matrix(si') ≈ (A + B + I)'

        # sums flatten:
        @test Matrix((op_a + I) + (op_b + 2 * I)) ≈ A + B + 3 * I
        @test length(((op_a + I) + (op_b + 2 * I)).terms) == 4
        @test Matrix((op_a + I) + op_b) ≈ A + B + I
        @test Matrix(op_a + (op_b + I)) ≈ A + B + I

        @test_throws ArgumentError MatrixShapedSum(())
        @test_throws DimensionMismatch MatrixShapedSum((op_a, asoperator(randn(Float32, 3, 3))))
        @test_throws DimensionMismatch op_a + asoperator(randn(Float32, 3, 3))

        @timed_testset "traits and definiteness" begin
            Σ1 = asoperator(A * A' + I, IsPosDef())
            Σ2 = asoperator(B * B', IsPosSemiDefOnly())
            @test PosDefNess(Σ1 + Σ2) === IsPosDef()
            @test PosDefNess(Σ2 + Σ2) === IsPosSemiDefOnly()
            @test Hermitianity(Σ1 + Σ2) === IsHermitian()
            @test isposdef(Σ1 + Σ2)

            l1 = asoperator(LowerTriangular(A))
            l2 = asoperator(Diagonal(randn(Float32, 5)))
            @test Triangularity(l1 + l2) === IsLowerTriangular()
            @test Triangularity(l2 + l2) === IsDiagonal()
            @test Symmetricity(l2 + l2) === IsSymmetric()

            # value-refined isposdef: all terms psd, at least one posdef:
            g = RowGramOperator(randn(Float32, 5, 7))
            m = g + I
            @test isposdef(m)
            @test ishermitian(m) && issymmetric(m)
        end

        @timed_testset "complex element types" begin
            Cs = randn(ComplexF64, 4, 4)
            Ds = randn(ComplexF64, 4, 4)
            sz = asoperator(Cs) + asoperator(Ds)
            @test Matrix(sz') ≈ (Cs + Ds)'
            @test Matrix(transpose(sz)) ≈ transpose(Cs + Ds)
            hz = asoperator(Hermitian(Cs * Cs' + I))
            @test ishermitian(hz + hz) && !issymmetric(hz + hz)
        end

        @timed_testset "vector terms" begin
            Ms = [randn(Float32, 5, 5) for _ in 1:20]
            ops = map(asoperator, Ms)
            sv = MatrixShapedSum(ops)
            @test sv isa MatrixShapedSum{Float32,<:AbstractVector}
            @test sv * x ≈ sum(Ms) * x
            @test sv * X ≈ sum(Ms) * X
            @test Matrix(sv') ≈ sum(Ms)'

            svi = sv + I
            @test svi isa MatrixShapedSum{Float32,<:AbstractVector}
            @test length(svi.terms) == 21
            @test svi * x ≈ (sum(Ms) + I) * x
            @test (op_a + sv).terms isa AbstractVector
            @test Matrix(op_a + sv) ≈ A + sum(Ms)
            @test Matrix(sv + sv) ≈ 2 * sum(Ms)
        end
    end

    @timed_testset "MatrixShapedProduct" begin
        A = randn(Float32, 4, 5)
        B = randn(Float32, 5, 6)
        op_a = asoperator(A)
        op_b = asoperator(B)
        x = randn(Float32, 6)
        X = randn(Float32, 6, 3)

        p = op_a * op_b
        @test p isa MatrixShapedProduct{Float32}
        @test p.factors == (op_a, op_b)
        @test size(p) == (4, 6)
        @test p * x ≈ A * (B * x)
        @test p * X ≈ A * (B * X)
        @test Matrix(p) ≈ A * B
        @test Matrix(p') ≈ (A * B)'
        @test p'.factors == (op_b', op_a')
        @test hash(p) == hash(asoperator(copy(A)) * op_b)
        @test p ≈ asoperator(copy(A)) * op_b
        @test occursin("MatrixShapedProduct", sprint(show, p))

        # products flatten:
        op_c = asoperator(randn(Float32, 6, 4))
        @test (p * op_c).factors == (op_a, op_b, op_c)
        @test (op_a * (op_b * op_c)).factors == (op_a, op_b, op_c)
        @test ((op_a * op_b) * (op_c * op_a)).factors == (op_a, op_b, op_c, op_a)

        # adjacent adjoint pairs stay ordinary products, Gram structure
        # is requested explicitly via rowgram_operator/colgram_operator:
        c = op_a' * op_a
        @test c isa MatrixShapedProduct
        @test Matrix(c) ≈ A' * A
        @test (op_b * op_b') isa MatrixShapedProduct

        @test_throws DimensionMismatch op_a * op_a
        @test_throws ArgumentError MatrixShapedProduct(())
        @test_throws DimensionMismatch MatrixShapedProduct((op_a, op_a))

        @timed_testset "traits" begin
            l = asoperator(LowerTriangular(randn(Float32, 5, 5)))
            d = asoperator(Diagonal(randn(Float32, 5)))
            @test Triangularity(l * d) === IsLowerTriangular()
            @test Triangularity(d * d) === IsDiagonal()
            @test Symmetricity(d * d) === IsSymmetric()

            q1 = asoperator(randn(Float32, 5, 5), IsUnitary())
            q2 = asoperator(randn(Float32, 5, 5), IsUnitary())
            @test Unitarity(q1 * q2) === IsUnitary()
            @test RowRankNess(q1 * q2) === IsFullRowRank()

            Σ = asoperator(randn(Float32, 5, 5), IsPosDef())
            @test PosDefNess(Σ * Σ) === UnknownPosDefNess()
        end

        @timed_testset "logabsdet" begin
            M1 = Diagonal([2.0, -3.0, 1.5])
            M2 = LowerTriangular(randn(3, 3) + 4 * I)
            p2 = asoperator(M1) * asoperator(Matrix(M2))
            lad = logabsdet(p2)
            lad_ref = logabsdet(Matrix(M1) * Matrix(M2))
            @test lad[1] ≈ lad_ref[1] && lad[2] ≈ lad_ref[2]
            @test_throws ArgumentError logabsdet(asoperator(randn(3, 4)) * asoperator(randn(4, 3)))
        end

        @timed_testset "inverse application" begin
            S1 = randn(5, 5) + 5 * I
            S2 = randn(5, 5) + 5 * I
            ps = asoperator(S1) * asoperator(S2)
            y5 = randn(5)
            @test ps \ y5 ≈ S2 \ (S1 \ y5)
            @test ps \ y5 ≈ Matrix(ps) \ y5

            # square products of non-square factors have no
            # factor-wise inverse:
            pr = op_a * op_b * asoperator(randn(Float32, 6, 4))
            @test_throws ArgumentError pr \ randn(Float32, 4)
        end

        @timed_testset "complex element types" begin
            Cs = randn(ComplexF64, 4, 4)
            Ds = randn(ComplexF64, 4, 4)
            pz = asoperator(Cs) * asoperator(Ds)
            @test Matrix(pz') ≈ (Cs * Ds)'
            @test Matrix(transpose(pz)) ≈ transpose(Cs * Ds)
        end

        @timed_testset "vector factors" begin
            Ms = [randn(Float32, 5, 5) for _ in 1:10]
            pv = MatrixShapedProduct(map(asoperator, Ms))
            @test pv isa MatrixShapedProduct{Float32,<:AbstractVector}
            z = randn(Float32, 5)
            @test pv * z ≈ foldl(*, Ms) * z
            @test (pv * asoperator(Ms[1])).factors isa AbstractVector
            @test Matrix(pv') ≈ foldl(*, Ms)'

            # structural verbs on vector storage:
            Msq = [randn(4, 4) + 4 * I for _ in 1:3]
            pvq = MatrixShapedProduct(map(asoperator, Msq))
            y4 = randn(4)
            @test pvq \ y4 ≈ Matrix(pvq) \ y4
            ladv = logabsdet(pvq)
            ladv_ref = logabsdet(Matrix(pvq))
            @test ladv[1] ≈ ladv_ref[1] && ladv[2] ≈ ladv_ref[2]
        end
    end
end
