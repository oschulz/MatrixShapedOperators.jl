# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).

module MatrixShapedOperatorsTestExt

using Test: @test, @testset

using LinearAlgebra

using MatrixShapedOperators
using MatrixShapedOperators: MatrixShapedOperator,
    Symmetricity, IsSymmetric, Hermitianity, IsHermitian,
    PosDefNess, IsPosDef, IsPosSemiDef, Triangularity, IsDiagonal,
    IsLowerTriangular, IsUpperTriangular, Unitarity, IsUnitary,
    RowRankNess, IsFullRowRank


function MatrixShapedOperators.test_operator(
    op::MatrixShapedOperator; directions::Symbol = :both, kwargs...
)
    directions === :both && return _test_operator(op, true; kwargs...)
    directions === :forward && return _test_operator(op, false; kwargs...)
    directions === :adjoint && return _test_operator(adjoint(op), false; kwargs...)
    throw(ArgumentError(
        "directions must be :both, :forward or :adjoint, got $(repr(directions))"
    ))
end

function _test_operator(
    op::MatrixShapedOperator{T}, test_adjoint::Bool;
    x::AbstractVector{<:Number} = _probe(T, size(op, 2), 1),
    y::AbstractVector{<:Number} = _probe(T, size(op, 2), 2),
    u::AbstractVector{<:Number} = _probe(T, size(op, 1), 3),
    compare = isapprox, materialize::Bool = true, kwargs...
) where T
    α, β = _probe_coeffs(float(T))
    @testset "test_operator: $(nameof(typeof(op))) of size $(size(op))" begin
        op_x = op * x
        op_y = op * y

        @testset "linearity" begin
            @test compare(op * (α .* x .+ β .* y), α .* op_x .+ β .* op_y; kwargs...)
        end

        if test_adjoint
            @testset "adjoint" begin
                @test compare(dot(u, op_x), dot(adjoint(op) * u, x); kwargs...)
            end
        end

        @testset "batched application" begin
            @test compare(op * hcat(x, y), hcat(op_x, op_y); kwargs...)
        end

        if materialize
            M = Matrix(op)

            @testset "materialization" begin
                @test compare(M * x, op_x; kwargs...)
            end

            @testset "declared traits" begin
                _test_traits(op, M, compare; kwargs...)
            end
        end
    end
    return nothing
end


function _test_traits(op::MatrixShapedOperator, M::AbstractMatrix, compare; kwargs...)
    m = size(M, 1)
    R = real(float(eltype(M)))

    if Symmetricity(op) isa IsSymmetric
        @test compare(M, transpose(M); kwargs...)
    end
    if Hermitianity(op) isa IsHermitian
        @test compare(M, adjoint(M); kwargs...)
    end

    pd = PosDefNess(op)
    if pd isa IsPosDef
        @test isposdef(Hermitian(M))
    elseif pd isa IsPosSemiDef && m > 0
        @test eigmin(Hermitian(M)) >= -sqrt(eps(R)) * max(norm(M), one(R))
    end

    tri = Triangularity(op)
    if tri isa IsDiagonal
        @test compare(M, Matrix(Diagonal(M)); kwargs...)
    elseif tri isa IsLowerTriangular
        @test compare(M, tril(M); kwargs...)
    elseif tri isa IsUpperTriangular
        @test compare(M, triu(M); kwargs...)
    end

    if Unitarity(op) isa IsUnitary
        @test compare(adjoint(M) * M, I; kwargs...)
    end
    if RowRankNess(op) isa IsFullRowRank
        @test rank(M) == m
    end

    return nothing
end


# Deterministic probe arrays, so that failures reproduce; complex
# coefficients ensure that real-linear but not complex-linear actions
# fail the linearity test:

_probe(::Type{T}, n::Integer, k::Integer) where {T<:Number} =
    [_probe_elem(float(T), i + 4 * k) for i in 1:n]

_probe_elem(::Type{R}, j::Integer) where {R<:Real} = R(cospi(j / 5))
_probe_elem(::Type{R}, j::Integer) where {R<:Complex} = R(cospi(j / 5), sinpi(j / 3))

_probe_coeffs(::Type{R}) where {R<:Real} = (R(3) / 4, R(-5) / 4)
_probe_coeffs(::Type{R}) where {R<:Complex} = (R(3, 1) / 4, R(-5, 2) / 4)


end # module MatrixShapedOperatorsTestExt
