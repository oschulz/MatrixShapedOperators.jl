# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).

using MatrixShapedOperators
using Test
using TimerOutputs: @timed_testset

using LinearAlgebra

@timed_testset "cholesky_factor" begin
    B = randn(5, 5)
    Σ = Matrix(Symmetric(B * B' + I))

    L = lower_cholesky(Σ)
    @test L isa LowerTriangular
    @test L * L' ≈ Σ
    @test all(>(0), diag(L))
    @test lower_cholesky(Symmetric(Σ)) ≈ L
    @test lower_cholesky(cholesky(Σ)) ≈ L

    D = Diagonal([4.0, 9.0, 16.0])
    LD = lower_cholesky(D)
    @test LD isa Diagonal
    @test LD == Diagonal([2.0, 3.0, 4.0])

    # the upper factor is the adjoint of the lower one:
    U = upper_cholesky(Σ)
    @test U isa UpperTriangular
    @test U == lower_cholesky(Σ)'
    @test U' * U ≈ Σ
    uop = upper_cholesky(asoperator(D))
    @test uop isa MatrixAsOperator
    @test parent(uop) == Diagonal([2.0, 3.0, 4.0])
    @test upper_cholesky(RowGramOperator(LowerTriangular(randn(3, 3) + 4 * I))) isa MatrixAsOperator
    @test upper_cholesky(4.0 * 𝟙(2)) == 2.0 * 𝟙(2)
    Ug = UpperTriangular(randn(3, 3) + 5 * I)
    gcu = ColGramOperator(Ug)
    Ucf = upper_cholesky(gcu)
    @test Matrix(Ucf)' * Matrix(Ucf) ≈ Matrix(gcu)
    bdc = BlockDiagOperator((asoperator(D), 4.0 * 𝟙(2)))
    Ub = upper_cholesky(bdc)
    @test Matrix(Ub)' * Matrix(Ub) ≈ Matrix(bdc)

    @timed_testset "complex" begin
        C = randn(ComplexF64, 4, 4)
        H = Matrix(Hermitian(C * C' + I))
        Lz = lower_cholesky(H)
        @test Lz * Lz' ≈ H
        @test all(x -> isreal(x) && real(x) > 0, diag(Lz))

        # complex-stored diagonals and uniform scalings must be real
        # non-negative to have a Cholesky factor:
        @test lower_cholesky(Diagonal(ComplexF64[4.0, 9.0])) == Diagonal([2.0, 3.0])
        @test_throws PosDefException lower_cholesky(Diagonal([1.0 + 1.0im, 1.0 + 0.0im]))
        @test_throws PosDefException lower_cholesky(Diagonal([-1.0, 2.0]))
        @test lower_cholesky((4.0 + 0.0im) * 𝟙(2)) == 2.0 * 𝟙(2)
        @test_throws ArgumentError lower_cholesky((1.0 + 1.0im) * 𝟙(2))
    end
end
