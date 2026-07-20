# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).

using MatrixShapedOperators
using Test
using TimerOutputs: @timed_testset

using LinearAlgebra

using MatrixShapedOperators: Symmetricity, IsSymmetric, UnknownSymmetricity,
    Hermitianity, IsHermitian, UnknownHermitianity,
    PosDefNess, IsPosSemiDefOnly, UnknownPosDefNess,
    Triangularity, IsDiagonal, UnknownTriangularity

@timed_testset "zero_operator" begin
    z = ZeroOperator(3, 2)
    x = randn(2)
    X = randn(2, 4)

    @test z isa ZeroOperator
    @test eltype(z) == Bool
    @test size(z) == (3, 2)
    @test z * x == zeros(3)
    @test z * x isa Vector{Float64}
    @test z * X == zeros(3, 4)
    @test z' isa ZeroOperator
    @test size(z') == (2, 3)
    @test conj(z) === z
    @test Matrix(z) == zeros(3, 2)
    @test iszero(z)
    @test z == ZeroOperator(3, 2)
    @test z != ZeroOperator(2, 3)
    @test hash(z) == hash(ZeroOperator(3, 2))
    @test z ≈ ZeroOperator(3, 2)
    @test occursin("𝟘(3, 2)", sprint(show, z))
    @test_throws ArgumentError ZeroOperator(-1, 2)

    op = asoperator(randn(Float32, 3, 3))
    @test zero(op) === ZeroOperator(3, 3)
    @test iszero(zero(op))

    @timed_testset "algebra collapses" begin
        sq = ZeroOperator(3, 3)
        @test op + sq === op
        @test sq + op === op
        @test sq + sq === sq
        @test (op + op) + sq isa MatrixShapedSum
        @test op * sq === ZeroOperator(3, 3)
        @test sq * op === ZeroOperator(3, 3)
        @test z * ZeroOperator(2, 5) === ZeroOperator(3, 5)
        @test (op * op) * sq === ZeroOperator(3, 3)
        @test sq * (op * op) === ZeroOperator(3, 3)
        @test 𝟙(3) * sq === ZeroOperator(3, 3)
        @test sq * 𝟙(3) === ZeroOperator(3, 3)
        # scalar scaling of the zero operator is absorbed:
        @test 2.5 * sq === sq
        @test -sq isa ZeroOperator
        @test Matrix(sq + 𝟙) == Matrix(I, 3, 3)
        @test_throws DimensionMismatch op + z
        @test_throws DimensionMismatch z * op
    end

    @timed_testset "traits and structural operations" begin
        sq = ZeroOperator(2, 2)
        @test Symmetricity(sq) === IsSymmetric()
        @test Hermitianity(sq) === IsHermitian()
        @test PosDefNess(sq) === IsPosSemiDefOnly()
        @test Triangularity(sq) === IsDiagonal()
        @test issymmetric(sq) && ishermitian(sq) && !isposdef(sq)
        @test Symmetricity(z) === UnknownSymmetricity()
        @test Hermitianity(z) === UnknownHermitianity()
        @test PosDefNess(z) === UnknownPosDefNess()
        @test Triangularity(z) === UnknownTriangularity()

        @test lower_cholesky(sq) === sq
        @test rowgram_factor(sq) === sq
        @test colgram_factor(sq) === sq
        @test factorize(sq) === sq
        @test_throws DimensionMismatch lower_cholesky(z)

        lad = logabsdet(sq)
        @test lad[1] == -Inf && lad[2] == false
        lad0 = logabsdet(ZeroOperator(0, 0))
        @test lad0[1] == 0.0 && lad0[2] == true
        @test_throws DimensionMismatch logabsdet(z)

        y2 = randn(2)
        @test_throws SingularException sq \ y2
        y0 = randn(0)
        @test ZeroOperator(0, 0) \ y0 === y0
    end

    @timed_testset "sizeless 𝟘" begin
        A = randn(Float32, 3, 3)
        opA = asoperator(A)

        @test 𝟘 isa MatrixShapedOperators.Zero
        @test 𝟘(3, 2) === ZeroOperator(3, 2)
        @test 𝟘(3) === ZeroOperator(3, 3)
        @test eltype(𝟘(3)) == Bool

        @test opA + 𝟘 === opA
        @test 𝟘 + opA === opA
        @test opA - 𝟘 === opA
        @test Matrix(𝟘 - opA) ≈ -A
        @test opA * 𝟘 === zero(opA)
        @test 𝟘 * opA === zero(opA)
        @test asoperator(randn(2, 3)) * 𝟘 === 𝟘(2, 3)

        # matrices, numbers and uniform scalings combine like with 0 * I:
        @test A + 𝟘 == A
        @test 𝟘 + A == A
        @test A - 𝟘 == A
        @test 𝟘 - A == -A
        @test A * 𝟘 == zeros(Float32, 3, 3)
        @test 𝟘 * A == zeros(Float32, 3, 3)
        @test 𝟘 * randn(2) == zeros(2)
        @test 2.5 * 𝟘 == 0.0 * I
        @test 𝟘 * 2.5 == 0.0 * I
        @test I + 𝟘 == I
        @test 𝟘 - 2 * I == -2 * I
        @test 𝟘 + 𝟘 === 𝟘
        @test 𝟘 - 𝟘 === 𝟘
        @test 𝟘 * 𝟘 === 𝟘
        @test -𝟘 === 𝟘
        @test 𝟙 * 𝟘 === 𝟘
        @test 𝟘 * 𝟙 === 𝟘
        @test 𝟙 + 𝟘 == I
        @test 𝟘 - 𝟙 == -I
        @test iszero(𝟘)

        # structural operations close over 𝟘 as well:
        @test 𝟘' === 𝟘
        @test transpose(𝟘) === 𝟘
        @test conj(𝟘) === 𝟘
        @test lower_cholesky(𝟘) === 𝟘
        @test upper_cholesky(𝟘) === 𝟘
        @test rowgram_factor(𝟘) === 𝟘
        @test colgram_factor(𝟘) === 𝟘
        @test logabsdet(𝟘) == (-Inf, 0.0)
        @test issymmetric(𝟘) && ishermitian(𝟘) && !isposdef(𝟘)
        @test Symmetricity(𝟘) === IsSymmetric()
        @test PosDefNess(𝟘) === IsPosSemiDefOnly()
        @test_throws SingularException 𝟘 \ randn(3)
        @test occursin("𝟘", sprint(show, 𝟘))

        # a zero base yields a pure low-rank operator:
        B = randn(4, 2)
        D = Symmetric(Matrix(Diagonal([1.0, -0.5])))
        w = woodbury_operator(𝟘, B, D)
        @test w isa WoodburyOperator
        @test Matrix(w) ≈ B * D * B'
        @test ishermitian(w)
        @test Hermitianity(w) === IsHermitian()
    end
end
