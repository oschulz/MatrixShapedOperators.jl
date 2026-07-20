# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).

using MatrixShapedOperators
using Test
using TimerOutputs: @timed_testset

using LinearAlgebra
import Statistics

using MatrixShapedOperators: IsPosDef

@timed_testset "statistics" begin
    x = randn(Float32, 5)

    Ms = [randn(Float32, 5, 5) for _ in 1:20]
    ops = map(asoperator, Ms)

    m = Statistics.mean(ops)
    @test m isa ScaledOperator{Float32}
    @test m.op isa MatrixShapedSum{Float32}
    @test m.op.terms isa AbstractVector
    @test m * x ≈ Statistics.mean(Ms) * x
    @test Matrix(m) ≈ Statistics.mean(Ms)

    mt = Statistics.mean((ops[1], ops[2]))
    @test Matrix(mt) ≈ (Ms[1] + Ms[2]) / 2

    # Scalar scaling preserves hermitianity and definiteness:
    covs = [asoperator(Symmetric(M'M + I), IsPosDef()) for M in Ms]
    mc = Statistics.mean(covs)
    @test ishermitian(mc)
    @test isposdef(mc)

    # Bool and integer element types promote like the usual mean:
    @test Matrix(Statistics.mean((𝟙(3), 𝟙(3)))) == Matrix(I, 3, 3)
    @test Matrix(Statistics.mean((2 * 𝟙(3), 4 * 𝟙(3)))) == Matrix(3.0 * I, 3, 3)
    @test Matrix(Statistics.mean((asoperator([1 0; 0 1]), asoperator([2 0; 0 2])))) ==
        [1.5 0.0; 0.0 1.5]

    # complex hermitian operators keep hermitianity under the real
    # scalar mean weight:
    Cz = randn(ComplexF64, 3, 3)
    Hz = asoperator(Hermitian(Cz * Cz' + I), MatrixShapedOperators.IsPosDef())
    mz = Statistics.mean((Hz, Hz))
    @test ishermitian(mz) && !issymmetric(mz)
    @test isposdef(mz)
    @test Matrix(mz) ≈ Matrix(Hz)
end
