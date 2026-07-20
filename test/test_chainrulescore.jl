# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).

using MatrixShapedOperators
using Test
using TimerOutputs: @timed_testset

using LinearAlgebra
import ChainRulesCore
using ChainRulesCore: rrule, frule, NoTangent

import MatrixShapedOperators as MSOs

@timed_testset "chainrulescore" begin
    A = Diagonal(randn(4))
    op = asoperator(A)
    ts = MSOs.traitsof(op)

    function check_non_differentiable(f, args...)
        r = rrule(f, args...)
        r === nothing && return false
        y, pb = r
        y == f(args...) || return false
        return all(t -> t === NoTangent(), pb(NoTangent()))
    end

    @test check_non_differentiable(MSOs.Symmetricity, op)
    @test check_non_differentiable(MSOs.Hermitianity, op)
    @test check_non_differentiable(MSOs.PosDefNess, op)
    @test check_non_differentiable(MSOs.Triangularity, op)
    @test check_non_differentiable(MSOs.Unitarity, op)
    @test check_non_differentiable(MSOs.RowRankNess, op)
    @test check_non_differentiable(MSOs.BatchedMulStyle, op)
    @test check_non_differentiable(MSOs.NumberDomain, op)
    @test check_non_differentiable(MSOs.traitsof, op)
    @test check_non_differentiable(MSOs.traitset, MSOs.IsPosDef(), MSOs.IsDiagonal())
    @test check_non_differentiable(MSOs.gettrait, ts, MSOs.Triangularity)
    @test check_non_differentiable(MSOs.trait_category, MSOs.IsPosDef())
    @test check_non_differentiable(MSOs.unknowntrait, MSOs.PosDefNess)
    @test check_non_differentiable(MSOs.trait_add, MSOs.IsPosDef(), MSOs.IsPosDef())
    @test check_non_differentiable(MSOs.trait_mul, MSOs.IsDiagonal(), MSOs.IsDiagonal())
    @test check_non_differentiable(MSOs.trait_blockdiag, MSOs.IsPosDef(), MSOs.IsPosDef())
    @test check_non_differentiable(MSOs.trait_adjoint, MSOs.IsLowerTriangular())
    @test check_non_differentiable(MSOs.trait_scale, MSOs.IsPosDef())
    @test check_non_differentiable(MSOs.traitset_add, ts, ts)
    @test check_non_differentiable(MSOs.traitset_mul, ts, ts)
    @test check_non_differentiable(MSOs.traitset_blockdiag, ts, ts)
    @test check_non_differentiable(MSOs.traitset_adjoint, ts)
    @test check_non_differentiable(MSOs.traitset_scale, ts)
    @test check_non_differentiable(size, op)
    @test check_non_differentiable(size, op, 1)
    @test check_non_differentiable(issymmetric, op)
    @test check_non_differentiable(ishermitian, op)
    @test check_non_differentiable(isposdef, op)
end
