# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).

using MatrixShapedOperators
using Test
using TimerOutputs: @timed_testset

using LinearAlgebra

using MatrixShapedOperators: Symmetricity, IsSymmetric, Hermitianity, IsHermitian,
    PosDefNess, IsPosDef, Triangularity, IsDiagonal, Unitarity, IsUnitary,
    RowRankNess, IsFullRowRank

@timed_testset "identity_operator" begin
    id = 𝟙(4)
    x = randn(4)
    X = randn(4, 3)

    @test id isa IdentityOperator
    @test eltype(id) == Bool
    @test id == IdentityOperator(4)
    @test size(id) == (4, 4)
    @test id' === id
    @test id * x === x
    @test id * X === X
    @test id \ x === x
    @test Matrix(id) == Matrix(I, 4, 4)
    @test asmatrix(id) == Diagonal(ones(Bool, 4))
    @test hash(id) == hash(IdentityOperator(4))
    @test occursin("𝟙(4)", sprint(show, id))

    @test Symmetricity(id) === IsSymmetric()
    @test Hermitianity(id) === IsHermitian()
    @test PosDefNess(id) === IsPosDef()
    @test Triangularity(id) === IsDiagonal()
    @test Unitarity(id) === IsUnitary()
    @test RowRankNess(id) === IsFullRowRank()
    @test issymmetric(id) && ishermitian(id) && isposdef(id)

    @test lower_cholesky(id) === id
    @test rowgram_factor(id) === id
    lad = logabsdet(id)
    @test lad[1] == 0.0 && lad[2] == true

    @test_throws ArgumentError IdentityOperator(-1)
    @test_throws ArgumentError 𝟙(-1)

    @timed_testset "sizeless 𝟙" begin
        A = randn(Float32, 4, 4)
        op = asoperator(A)

        s = op + 𝟙
        @test s isa MatrixShapedSum{Float32}
        @test s.terms[2] isa IdentityOperator
        @test s * x ≈ A * x .+ x
        @test Matrix(𝟙 + op) ≈ A + I
        @test Matrix(op - 𝟙) ≈ A - I
        @test Matrix(𝟙 - op) ≈ I - A

        # multiplying with the identity is a type-level no-op:
        @test op * 𝟙 === op
        @test 𝟙 * op === op
        @test op * 𝟙(4) === op
        @test 𝟙(4) * op === op
        @test 𝟙(4) * 𝟙(4) === 𝟙(4)

        # scaling the sized identity yields a scaled identity, which
        # keeps the structural operations available:
        @test 2.5 * 𝟙(4) === ScaledOperator(2.5, 𝟙(4))
        @test 2.5 * 𝟙(4) isa UniformScalingOperator{Float64}
        @test 𝟙(4) * 2.5 === ScaledOperator(2.5, 𝟙(4))
        @test 2 * (2.5 * 𝟙(4)) === ScaledOperator(5.0, 𝟙(4))
        @test lower_cholesky(2.5 * 𝟙(4)) == sqrt(2.5) * 𝟙(4)
        @test rowgram_factor(2.5 * 𝟙(4)) == sqrt(2.5) * 𝟙(4)
        @test (2.5 * 𝟙(4)) \ x ≈ x ./ 2.5
        @test first(logabsdet(2.5 * 𝟙(4))) ≈ 4 * log(2.5)
        @test (op * asoperator(A)) * 𝟙(4) isa MatrixShapedProduct

        # a scaled identity factor melts into a scalar scaling:
        @test (2.5 * 𝟙(4)) * op === ScaledOperator(2.5, op)
        @test op * (2.5 * 𝟙(4)) === ScaledOperator(2.5, op)
        @test (2.5 * 𝟙(4)) * (2.0 * 𝟙(4)) === ScaledOperator(5.0, 𝟙(4))
        @test (2.5 * 𝟙(4)) * 𝟙(4) === 2.5 * 𝟙(4)
        @test 𝟙(4) * (2.5 * 𝟙(4)) === 2.5 * 𝟙(4)
        p = op * asoperator(A)
        @test p * (2.5 * 𝟙(4)) === ScaledOperator(2.5, p)
        @test (2.5 * 𝟙(4)) * p === ScaledOperator(2.5, p)
        @test (2.5 * 𝟙(4)) * ZeroOperator(4, 2) == ZeroOperator(4, 2)
        @test ZeroOperator(2, 4) * (2.5 * 𝟙(4)) == ZeroOperator(2, 4)

        # solving against identities and scaled identities is exact:
        @test 𝟙(4) \ op === op
        @test Matrix((2.0 * 𝟙(4)) \ op) ≈ A ./ 2
        # a zero scaling of a non-empty operator is singular, like a
        # zero matrix, the empty solve stays exact:
        @test_throws SingularException (0.0 * 𝟙(4)) \ x
        @test_throws SingularException (0.0 * asoperator(A)) \ x
        @test_throws SingularException (0.0 * 𝟙(4)) \ op
        @test (0.0 * 𝟙(0)) \ Float64[] == Float64[]
        @test (0.0 * 𝟙(0)) \ 𝟙(0) === 𝟙(0)
        # conjugation keeps the scaled-identity form:
        @test conj((1.0 + 2.0im) * 𝟙(4)) === (1.0 - 2.0im) * 𝟙(4)
        @test logabsdet(𝟙(4)) === (0.0, true)
        @test factorize(𝟙(4)) === 𝟙(4)
        @test factorize(2.5 * 𝟙(4)) === 2.5 * 𝟙(4)
        @test 𝟙 * [1.0, 2.0] == [1.0, 2.0]
        @test 𝟙 / 2 === I / 2
        @test 𝟙' === 𝟙
        @test 𝟙 isa MatrixShapedOperators.One
        @test occursin("𝟙", sprint(show, 𝟙))

        @test_throws DimensionMismatch asoperator(randn(2, 3)) + 𝟙
        @test_throws DimensionMismatch op * 𝟙(3)

        # matrices, numbers and uniform scalings combine like with I:
        @test A + 𝟙 == A + I
        @test 𝟙 + A == I + A
        @test A - 𝟙 == A - I
        @test 𝟙 - A == I - A
        @test A * 𝟙 == A
        @test 𝟙 * A == A
        @test 2.5 * 𝟙 === 2.5 * I
        @test 𝟙 * 2.5 === 2.5 * I
        @test 2 * I + 𝟙 === 3 * I
        @test 𝟙 - 2 * I === -1 * I
        @test (2 * I) * 𝟙 === 2 * I
        @test 𝟙 + 𝟙 === 2 * I
        @test 𝟙 - 𝟙 === 0 * I
        @test 𝟙 * 𝟙 === 𝟙
        @test -𝟙 === -1 * I

        # structural operations close over 𝟙 as well:
        @test transpose(𝟙) === 𝟙
        @test conj(𝟙) === 𝟙
        @test lower_cholesky(𝟙) === 𝟙
        @test upper_cholesky(𝟙) === 𝟙
        @test rowgram_factor(𝟙) === 𝟙
        @test colgram_factor(𝟙) === 𝟙
        lad1 = logabsdet(𝟙)
        @test lad1[1] == 0 && lad1[2] == 1
        y1 = randn(4)
        @test 𝟙 \ y1 === y1
        @test 𝟙 \ op === op
        @test issymmetric(𝟙) && ishermitian(𝟙) && isposdef(𝟙)
        @test Symmetricity(𝟙) === IsSymmetric()
        @test Hermitianity(𝟙) === IsHermitian()
        @test PosDefNess(𝟙) === IsPosDef()
        @test Triangularity(𝟙) === IsDiagonal()
        @test Unitarity(𝟙) === IsUnitary()
        @test RowRankNess(𝟙) === IsFullRowRank()
        @test MatrixShapedOperators.traitsof(𝟙) === MatrixShapedOperators.traitsof(𝟙(4))

        # a structurally posdef identity term makes a psd-plus-identity
        # metric positive definite in the type domain:
        g = rowgram_operator(randn(Float32, 4, 6))
        @test PosDefNess(g + 𝟙) === IsPosDef()
        @test isposdef(g + 𝟙)
    end

    @timed_testset "blockdiag collapse" begin
        bd = blockdiag_operator(diagonal_operator([2.0, 3.0]), 𝟙(2))
        @test bd isa MatrixAsOperator{Float64,<:Tuple,<:Diagonal}
        @test parent(bd) == Diagonal([2.0, 3.0, 1.0, 1.0])
    end
end
