# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).

using MatrixShapedOperators
using Test
using TimerOutputs: @timed_testset

using LinearAlgebra
import Adapt

import MatrixShapedOperators as MSOs

# Adaptor that converts array storage to 32-bit precision:
struct _F32Adaptor end
Adapt.adapt_storage(::_F32Adaptor, x::AbstractArray) = Float32.(x)
Adapt.adapt_storage(::_F32Adaptor, x::AbstractArray{<:Complex}) = ComplexF32.(x)

@timed_testset "adapt" begin
    to = _F32Adaptor()

    A = randn(4, 4)
    w = asoperator(A, MSOs.IsPosDef())
    wa = Adapt.adapt(to, w)
    @test wa isa MatrixAsOperator{Float32}
    @test parent(wa) ≈ Float32.(A)
    @test MSOs.PosDefNess(wa) === MSOs.IsPosDef()

    d = diagonal_operator(randn(4))
    da = Adapt.adapt(to, d)
    @test da isa MatrixAsOperator{Float32,<:Tuple,<:Diagonal}
    @test MSOs.Triangularity(da) === MSOs.IsDiagonal()

    g = RowGramOperator(randn(3, 5))
    ga = Adapt.adapt(to, g)
    @test ga isa RowGramOperator{Float32}
    @test Matrix(ga) ≈ Float32.(Matrix(g))

    wo = WoodburyOperator(Diagonal(rand(4) .+ 1), randn(4, 2), Symmetric(randn(2, 2)))
    woa = Adapt.adapt(to, wo)
    @test woa isa WoodburyOperator{Float32}
    @test Matrix(woa) ≈ Float32.(Matrix(wo))

    s = asoperator(A) + I
    sa = Adapt.adapt(to, s)
    @test sa isa MatrixShapedSum
    @test eltype(sa.terms[1]) == Float32
    @test sa.terms[2] isa UniformScalingOperator

    p = asoperator(A) * asoperator(randn(4, 4))
    pa = Adapt.adapt(to, p)
    @test pa isa MatrixShapedProduct
    @test Matrix(pa) ≈ Float32.(Matrix(p))

    u = 2.0 * 𝟙(4)
    @test Adapt.adapt(to, u) === u

    # scaled operators adapt through the scaling wrapper:
    sc = 2.0 * asoperator(A)
    sca = Adapt.adapt(to, sc)
    @test sca isa ScaledOperator
    @test parent(sca.op) ≈ Float32.(A)

    # conjugated operators adapt through the conjugation wrapper:
    Cz = randn(ComplexF64, 3, 3)
    ca = Adapt.adapt(to, conj(asoperator(Cz)))
    @test ca isa MatrixShapedOperator{ComplexF32}
    @test Matrix(ca) ≈ ComplexF32.(conj(Cz))

    bd = blockdiag_operator(asoperator(randn(2, 3)), asoperator(randn(3, 2)))
    bda = Adapt.adapt(to, bd)
    @test bda isa BlockDiagOperator{Float32}
    @test Matrix(bda) ≈ Float32.(Matrix(bd))

    # the scalar of a scaled operator follows the storage precision:
    sca2 = Adapt.adapt(to, 2.5 * asoperator(randn(3, 3)))
    @test eltype(sca2) == Float32
    @test sca2.s isa Float32
    # no storage, the scalar keeps its type:
    @test Adapt.adapt(to, 2.5 * 𝟙(3)) === 2.5 * 𝟙(3)

    # factorizations adapt through their cached factor:
    wf = factorize(woodbury_operator(Diagonal(rand(3) .+ 1), randn(3, 0), Symmetric(zeros(0, 0))))
    wfa = Adapt.adapt(to, wf)
    @test wfa isa WoodburyFactorization
    @test eltype(wfa.F) == Float32
end
