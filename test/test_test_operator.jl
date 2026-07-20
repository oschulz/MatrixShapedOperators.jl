# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).

using MatrixShapedOperators
using Test
using TimerOutputs: @timed_testset

using LinearAlgebra

using MatrixShapedOperators: test_operator, IsHermitian, IsPosDef

# Records test results instead of throwing or reporting them, to check
# that broken operators are reported as failures:
struct _RecordingTestSet <: Test.AbstractTestSet
    results::Vector{Any}
    _RecordingTestSet(::String) = new([])
end

Test.record(ts::_RecordingTestSet, t) = (push!(ts.results, t); t)

# Nested testsets inherit the type, they attach to their parent as
# usual; the outermost one does not, so that failures stay contained:
function Test.finish(ts::_RecordingTestSet)
    parent = Test.get_testset_depth() > 0 ? Test.get_testset() : nothing
    parent isa _RecordingTestSet && Test.record(parent, ts)
    return ts
end

_n_failed(::Any) = 0
_n_failed(::Union{Test.Fail,Test.Error}) = 1
_n_failed(ts::Test.DefaultTestSet) = _n_failed(ts.results)
_n_failed(ts::_RecordingTestSet) = _n_failed(ts.results)
_n_failed(rs::AbstractVector) = sum(_n_failed, rs; init = 0)

_failures_of(op; kwargs...) = _n_failed(
    Test.@testset _RecordingTestSet "recorded" begin
        test_operator(op; kwargs...)
    end
)

@timed_testset "test_operator" begin
    A = [1.0 2.0 -0.5; 0.5 -1.0 2.0; 2.0 0.5 1.0; -1.0 1.5 0.5]

    @test isnothing(test_operator(asoperator(A)))
    @test isnothing(test_operator(asoperator(A); materialize = false))
    @test isnothing(test_operator(asoperator(A); rtol = 1e-10))

    S = A' * A + 5 * I
    test_operator(asoperator(Symmetric(S), IsPosDef()))
    test_operator(asoperator(LowerTriangular(S)))
    test_operator(𝟙(4))
    test_operator(ZeroOperator(3, 2))
    test_operator(rowgram_operator(A))
    test_operator(colgram_operator(A))
    test_operator(blockdiag_operator(asoperator(A), 𝟙(2)))
    test_operator(woodbury_operator(Diagonal([1.0, 2.0, 3.0]), reshape([1.0, 0.5, -1.0], 3, 1), Symmetric(fill(2.0, 1, 1))))
    test_operator(asoperator(qr(A).Q))
    test_operator(2.5 * asoperator(A))
    test_operator(asoperator(A) * asoperator(A'))

    # user-supplied probe arrays:
    test_operator(asoperator(A); x = [1.0, -2.0, 0.5], y = [0.5, 1.0, 2.0], u = [1.0, 0.0, -1.0, 2.0])

    @timed_testset "complex" begin
        Az = [1.0+1.0im 2.0; -0.5im 1.5+0.5im]
        test_operator(asoperator(Az))
        test_operator(asoperator(Az * Az' + 3 * I, IsPosDef()))
    end

    @timed_testset "one-directional operators" begin
        fwd_only = mulfunc_operator(Float64, (4, 3), z -> A * z, nothing)
        adj_only = mulfunc_operator(Float64, (4, 3), nothing, z -> A' * z)

        @test isnothing(test_operator(fwd_only; directions = :forward))
        @test isnothing(test_operator(adj_only; directions = :adjoint))
        # double adjoint restores the forward direction:
        @test isnothing(test_operator(adj_only'; directions = :forward))
        # both directions still test the adjoint identity by default:
        @test isnothing(test_operator(asoperator(A); directions = :both))
        @test _failures_of(mulfunc_operator(Float64, (4, 3), z -> A * z, z -> 2 .* (A' * z))) >
            _failures_of(mulfunc_operator(Float64, (4, 3), z -> A * z, z -> 2 .* (A' * z)); directions = :forward)

        @test_throws ArgumentError test_operator(fwd_only; directions = :sideways)
    end

    @timed_testset "broken operators are reported" begin
        # an inconsistent adjoint:
        mf_bad = mulfunc_operator(Float64, (4, 3), z -> A * z, z -> 2 .* (A' * z))
        @test _failures_of(mf_bad) > 0
        # a real-linear but not complex-linear action:
        conj_bad = mulfunc_operator(ComplexF64, (3, 3), z -> conj.(z), z -> conj.(z))
        @test _failures_of(conj_bad) > 0
        # a falsely declared trait, caught by materialization only:
        herm_bad = asoperator(A' * A + [0.0 1.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0], IsHermitian())
        @test _failures_of(herm_bad) > 0
        @test _failures_of(herm_bad; materialize = false) == 0
    end
end
