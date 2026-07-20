# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).

# Must be set before Reactant loads XLA, which otherwise grabs most GPU memory:
haskey(ENV, "XLA_REACTANT_GPU_MEM_FRACTION") || (ENV["XLA_REACTANT_GPU_MEM_FRACTION"] = "0.4")

using MatrixShapedOperators
using Test
using TimerOutputs: @timed_testset

using LinearAlgebra
import Reactant

using MatrixShapedOperators: IsHermitian, IsPosDef

@timed_testset "reactant" begin
    A = randn(Float32, 4, 4)
    B = randn(Float32, 4, 2)
    a = rand(Float32, 4) .+ 1
    D = Float32[2 1; 1 3]
    x = randn(Float32, 4)
    X = randn(Float32, 4, 3)

    rA = Reactant.to_rarray(A)
    rB = Reactant.to_rarray(B)
    ra = Reactant.to_rarray(a)
    rD = Reactant.to_rarray(D)
    rx = Reactant.to_rarray(x)
    rX = Reactant.to_rarray(X)

    f_wrap(A, x) = asoperator(A) * x
    @test Array(Reactant.@jit f_wrap(rA, rx)) ≈ A * x

    f_algebra(A, a, x) = (2 * asoperator(A) + diagonal_operator(a) + I) * x
    @test Array(Reactant.@jit f_algebra(rA, ra, rx)) ≈ (2 * A + Diagonal(a) + I) * x

    f_product(A, x) = (asoperator(A)' * asoperator(A)) * x
    @test Array(Reactant.@jit f_product(rA, rx)) ≈ A' * (A * x)

    f_gram_batch(A, X) = RowGramOperator(A) * X
    @test Array(Reactant.@jit f_gram_batch(rA, rX)) ≈ A * (A' * X)

    # Traced construction requires trait-declared components instead of
    # value checks (ishermitian on traced arrays is not traceable):
    f_woodbury(a, B, D, x) =
        WoodburyOperator(diagonal_operator(a), B, asoperator(D, IsHermitian())) * x
    @test Array(Reactant.@jit f_woodbury(ra, rB, rD, rx)) ≈ Diagonal(a) * x + B * (D * (B' * x))

    f_blockdiag(A, a, x) = blockdiag_operator(asoperator(A), diagonal_operator(a)) * x
    x8 = randn(Float32, 8)
    rx8 = Reactant.to_rarray(x8)
    ref = [A * x8[1:4]; a .* x8[5:8]]
    @test Array(Reactant.@jit f_blockdiag(rA, ra, rx8)) ≈ ref

    # the sizeless identity and zero are type-level and trace-safe:
    f_sizeless(A, x) = (asoperator(A) + 𝟙) * x + (asoperator(A) + 𝟘) * x
    @test Array(Reactant.@jit f_sizeless(rA, rx)) ≈ (A + I) * x + A * x

    # one-directional function operators construct and apply under
    # tracing, including through the adjoint:
    f_fwd_only(A, x) = mulfunc_operator(Float32, (4, 4), Base.Fix1(*, A), nothing) * x
    @test Array(Reactant.@jit f_fwd_only(rA, rx)) ≈ A * x
    f_adj_only(A, x) = adjoint(mulfunc_operator(Float32, (4, 4), nothing, Base.Fix1(*, A))) * x
    @test Array(Reactant.@jit f_adj_only(rA, rx)) ≈ A * x

    # zero scaling stays a plain scaling under application:
    f_zero_scale(A, x) = (0.0f0 * asoperator(A)) * x
    @test Array(Reactant.@jit f_zero_scale(rA, rx)) ≈ zero(x)

    # lazy conjugation traces through complex application:
    Az = randn(ComplexF32, 4, 4)
    z = randn(ComplexF32, 4)
    rAz = Reactant.to_rarray(Az)
    rz = Reactant.to_rarray(z)
    f_conj(A, z) = conj(asoperator(A)) * z
    @test Array(Reactant.@jit f_conj(rAz, rz)) ≈ conj(Az) * z

    # the all-diagonal block collapse mixes device-backed diagonals
    # with the host-constant diagonal of a scaled identity:
    f_diag_collapse(a, x) = blockdiag_operator(diagonal_operator(a), 2.0f0 * 𝟙(4)) * x
    @test Array(Reactant.@jit f_diag_collapse(ra, rx8)) ≈ [a .* x8[1:4]; 2.0f0 .* x8[5:8]]
end
