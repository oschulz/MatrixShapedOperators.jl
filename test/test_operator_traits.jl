# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).

using MatrixShapedOperators
using Test
using TimerOutputs: @timed_testset

using LinearAlgebra

using MatrixShapedOperators: Symmetricity, IsSymmetric, UnknownSymmetricity,
    Hermitianity, IsHermitian, UnknownHermitianity,
    PosDefNess, IsPosSemiDef, IsPosDef, IsPosSemiDefOnly, UnknownPosDefNess,
    Triangularity, IsDiagonal, IsLowerTriangular, IsUpperTriangular,
    IsLowerTriangularOrDiagonal, IsUpperTriangularOrDiagonal, UnknownTriangularity,
    Unitarity, IsUnitary, UnknownUnitarity,
    RowRankNess, IsFullRowRank, UnknownRowRankNess,
    BatchedMulStyle, BatchedMul, ColumnwiseMul,
    NumberDomain, traitset, gettrait, traitsof, trait_category, unknowntrait,
    trait_add, trait_mul, trait_blockdiag, trait_adjoint, trait_scale,
    traitset_add, traitset_mul, traitset_blockdiag, traitset_adjoint, traitset_scale

# Third-party trait category, defined outside of MatrixShapedOperators:
abstract type TestSparsity end
struct IsTestSparse <: TestSparsity end
struct UnknownTestSparsity <: TestSparsity end
MatrixShapedOperators.unknowntrait(::Type{TestSparsity}) = UnknownTestSparsity()

@timed_testset "operator_traits" begin
    @timed_testset "NumberDomain" begin
        @test @inferred(NumberDomain(Float64)) === Real
        @test @inferred(NumberDomain(ComplexF64)) === Complex
        @test NumberDomain(Float32) === Real
        @test NumberDomain(Int) === Real
        @test NumberDomain(randn(3)) === Real
        @test NumberDomain(randn(ComplexF32, 3, 3)) === Complex
    end

    @timed_testset "trait_category and unknowntrait" begin
        @test trait_category(IsSymmetric()) === Symmetricity
        @test trait_category(IsHermitian()) === Hermitianity
        @test trait_category(IsPosDef()) === PosDefNess
        @test trait_category(IsPosSemiDefOnly()) === PosDefNess
        @test trait_category(UnknownTriangularity()) === Triangularity
        @test trait_category(IsUnitary()) === Unitarity
        @test trait_category(IsFullRowRank()) === RowRankNess
        @test trait_category(IsTestSparse()) === TestSparsity

        @test unknowntrait(Symmetricity) === UnknownSymmetricity()
        @test unknowntrait(Hermitianity) === UnknownHermitianity()
        @test unknowntrait(PosDefNess) === UnknownPosDefNess()
        @test unknowntrait(Triangularity) === UnknownTriangularity()
        @test unknowntrait(Unitarity) === UnknownUnitarity()
        @test unknowntrait(RowRankNess) === UnknownRowRankNess()
        @test unknowntrait(TestSparsity) === UnknownTestSparsity()
    end

    @timed_testset "traitset" begin
        @test @inferred(traitset()) === ()
        @test @inferred(traitset(IsSymmetric())) === (IsSymmetric(),)
        @test traitset(UnknownSymmetricity(), UnknownPosDefNess()) === ()

        # canonical order (by category name) and order invariance:
        @test traitset(IsSymmetric(), IsHermitian()) === (IsHermitian(), IsSymmetric())
        @test traitset(IsHermitian(), IsSymmetric()) === traitset(IsSymmetric(), IsHermitian())

        # first non-Unknown declaration per category wins, Unknowns never block:
        @test traitset(IsPosSemiDefOnly(), IsPosDef()) === traitset(IsPosSemiDefOnly())
        @test traitset(UnknownPosDefNess(), IsPosDef()) === traitset(IsPosDef())

        # implication closure:
        @test gettrait(traitset(IsPosDef()), RowRankNess) === IsFullRowRank()
        @test gettrait(traitset(IsPosDef()), Hermitianity) === IsHermitian()
        @test gettrait(traitset(IsPosSemiDefOnly()), Hermitianity) === IsHermitian()
        @test gettrait(traitset(IsPosSemiDefOnly()), RowRankNess) === UnknownRowRankNess()
        @test gettrait(traitset(IsUnitary()), RowRankNess) === IsFullRowRank()
        @test gettrait(traitset(IsDiagonal()), Symmetricity) === IsSymmetric()
    end

    @timed_testset "gettrait" begin
        ts = traitset(IsPosDef(), IsLowerTriangular())
        @test @inferred(gettrait(ts, PosDefNess)) === IsPosDef()
        @test gettrait(ts, PosDefNess) isa IsPosSemiDef
        @test gettrait(ts, Triangularity) === IsLowerTriangular()
        @test gettrait(ts, Triangularity) isa IsLowerTriangularOrDiagonal
        @test gettrait(ts, Symmetricity) === UnknownSymmetricity()
        @test gettrait(typeof(ts), Triangularity) === IsLowerTriangular()
        @test gettrait((), Unitarity) === UnknownUnitarity()

        # lookup is invariant to the tuple parameter order:
        @test gettrait(Tuple{IsSymmetric,IsHermitian}, Hermitianity) === IsHermitian()
        @test gettrait(Tuple{IsHermitian,IsSymmetric}, Hermitianity) === IsHermitian()
        @test gettrait(Tuple{IsDiagonal}, Triangularity) isa IsUpperTriangularOrDiagonal
    end

    @timed_testset "third-party trait category" begin
        ts = traitset(IsTestSparse(), IsSymmetric())
        @test ts === traitset(IsSymmetric(), IsTestSparse())
        @test gettrait(ts, TestSparsity) === IsTestSparse()
        @test gettrait(ts, Hermitianity) === UnknownHermitianity()
        @test traitset(UnknownTestSparsity()) === ()
        # conservative algebra fallbacks apply:
        @test trait_add(IsTestSparse(), IsTestSparse()) === UnknownTestSparsity()
        @test trait_adjoint(IsTestSparse()) === UnknownTestSparsity()
    end

    @timed_testset "trait algebra" begin
        @test trait_add(IsSymmetric(), IsSymmetric()) === IsSymmetric()
        @test trait_add(IsSymmetric(), UnknownSymmetricity()) === UnknownSymmetricity()
        @test trait_add(IsHermitian(), IsHermitian()) === IsHermitian()
        @test trait_add(IsPosDef(), IsPosDef()) === IsPosDef()
        @test trait_add(IsPosDef(), IsPosSemiDefOnly()) === IsPosDef()
        @test trait_add(IsPosSemiDefOnly(), IsPosDef()) === IsPosDef()
        @test trait_add(IsPosSemiDefOnly(), IsPosSemiDefOnly()) === IsPosSemiDefOnly()
        @test trait_add(IsPosDef(), UnknownPosDefNess()) === UnknownPosDefNess()
        @test trait_add(IsDiagonal(), IsDiagonal()) === IsDiagonal()
        @test trait_add(IsLowerTriangular(), IsDiagonal()) === IsLowerTriangular()
        @test trait_add(IsUpperTriangular(), IsDiagonal()) === IsUpperTriangular()
        @test trait_add(IsLowerTriangular(), IsUpperTriangular()) === UnknownTriangularity()
        @test trait_add(IsUnitary(), IsUnitary()) === UnknownUnitarity()
        @test trait_add(IsFullRowRank(), IsFullRowRank()) === UnknownRowRankNess()

        @test trait_mul(IsDiagonal(), IsDiagonal()) === IsDiagonal()
        @test trait_mul(IsLowerTriangular(), IsDiagonal()) === IsLowerTriangular()
        @test trait_mul(IsDiagonal(), IsUpperTriangular()) === IsUpperTriangular()
        @test trait_mul(IsLowerTriangular(), IsUpperTriangular()) === UnknownTriangularity()
        @test trait_mul(IsUnitary(), IsUnitary()) === IsUnitary()
        @test trait_mul(IsFullRowRank(), IsFullRowRank()) === IsFullRowRank()
        @test trait_mul(IsSymmetric(), IsSymmetric()) === UnknownSymmetricity()
        @test trait_mul(IsPosDef(), IsPosDef()) === UnknownPosDefNess()

        @test trait_blockdiag(IsSymmetric(), IsSymmetric()) === IsSymmetric()
        @test trait_blockdiag(IsHermitian(), IsHermitian()) === IsHermitian()
        @test trait_blockdiag(IsPosDef(), IsPosDef()) === IsPosDef()
        @test trait_blockdiag(IsPosDef(), IsPosSemiDefOnly()) === IsPosSemiDefOnly()
        @test trait_blockdiag(IsUnitary(), IsUnitary()) === IsUnitary()
        @test trait_blockdiag(IsFullRowRank(), IsFullRowRank()) === IsFullRowRank()

        @test trait_adjoint(IsSymmetric()) === IsSymmetric()
        @test trait_adjoint(IsHermitian()) === IsHermitian()
        @test trait_adjoint(IsPosDef()) === IsPosDef()
        @test trait_adjoint(IsDiagonal()) === IsDiagonal()
        @test trait_adjoint(IsLowerTriangular()) === IsUpperTriangular()
        @test trait_adjoint(IsUpperTriangular()) === IsLowerTriangular()
        @test trait_adjoint(IsUnitary()) === IsUnitary()

        @test trait_scale(IsSymmetric()) === IsSymmetric()
        @test trait_scale(IsHermitian()) === IsHermitian()
        @test trait_scale(IsPosDef()) === UnknownPosDefNess()
        @test trait_scale(IsLowerTriangular()) === IsLowerTriangular()
        @test trait_scale(IsUnitary()) === UnknownUnitarity()
        @test trait_scale(IsFullRowRank()) === UnknownRowRankNess()
    end

    @timed_testset "traitset algebra" begin
        ta = traitset(IsPosDef(), IsLowerTriangular())
        tb = traitset(IsPosSemiDefOnly(), IsDiagonal())

        tsum = @inferred(traitset_add(ta, tb))
        @test gettrait(tsum, PosDefNess) === IsPosDef()
        @test gettrait(tsum, Hermitianity) === IsHermitian()
        @test gettrait(tsum, Triangularity) === IsLowerTriangular()
        # re-closure: posdef implies full row rank even though addition
        # has no rank rule of its own:
        @test gettrait(tsum, RowRankNess) === IsFullRowRank()
        @test gettrait(tsum, Symmetricity) === UnknownSymmetricity()

        tprod = @inferred(traitset_mul(traitset(IsUnitary()), traitset(IsUnitary())))
        @test gettrait(tprod, Unitarity) === IsUnitary()
        @test gettrait(tprod, RowRankNess) === IsFullRowRank()
        @test gettrait(traitset_mul(ta, tb), Triangularity) === IsLowerTriangular()
        @test gettrait(traitset_mul(ta, tb), PosDefNess) === UnknownPosDefNess()

        tbd = traitset_blockdiag(ta, tb)
        @test gettrait(tbd, PosDefNess) === IsPosSemiDefOnly()
        @test gettrait(tbd, Hermitianity) === IsHermitian()
        @test gettrait(tbd, Triangularity) === UnknownTriangularity()

        tadj = @inferred(traitset_adjoint(ta))
        @test gettrait(tadj, Triangularity) === IsUpperTriangular()
        @test gettrait(tadj, PosDefNess) === IsPosDef()
        @test gettrait(tadj, RowRankNess) === IsFullRowRank()

        tsc = traitset_scale(ta)
        @test gettrait(tsc, Triangularity) === IsLowerTriangular()
        @test gettrait(tsc, PosDefNess) === UnknownPosDefNess()
        @test gettrait(tsc, Hermitianity) === IsHermitian()
    end

    @timed_testset "stdlib structure queries" begin
        A = randn(4, 4)
        @test Symmetricity(A) === UnknownSymmetricity()
        @test Symmetricity(Symmetric(A)) === IsSymmetric()
        @test Symmetricity(SymTridiagonal(randn(4), randn(3))) === IsSymmetric()
        @test Symmetricity(Diagonal(randn(4))) === IsSymmetric()
        @test Symmetricity(Hermitian(A)) === IsSymmetric()
        @test Symmetricity(Hermitian(randn(ComplexF64, 4, 4))) === UnknownSymmetricity()
        @test Hermitianity(A) === UnknownHermitianity()
        @test Hermitianity(Hermitian(A)) === IsHermitian()
        @test Hermitianity(Diagonal(randn(4))) === IsHermitian()
        @test Hermitianity(Diagonal(randn(ComplexF64, 4))) === UnknownHermitianity()
        @test Hermitianity(Symmetric(A)) === IsHermitian()
        @test Hermitianity(Symmetric(randn(ComplexF64, 4, 4))) === UnknownHermitianity()
        @test Hermitianity(SymTridiagonal(randn(4), randn(3))) === IsHermitian()
        @test PosDefNess(A) === UnknownPosDefNess()
        @test Triangularity(A) === UnknownTriangularity()
        @test Triangularity(LowerTriangular(A)) === IsLowerTriangular()
        @test Triangularity(UnitLowerTriangular(A)) === IsLowerTriangular()
        @test Triangularity(UpperTriangular(A)) === IsUpperTriangular()
        @test Triangularity(UnitUpperTriangular(A)) === IsUpperTriangular()
        @test Triangularity(Diagonal(randn(4))) === IsDiagonal()
        @test Triangularity(Bidiagonal(randn(4), randn(3), :L)) === IsLowerTriangular()
        @test Triangularity(Bidiagonal(randn(4), randn(3), :U)) === IsUpperTriangular()
        @test Unitarity(A) === UnknownUnitarity()
        @test Unitarity(qr(A).Q) === IsUnitary()
        @test RowRankNess(A) === UnknownRowRankNess()
        @test RowRankNess(qr(A).Q) === IsFullRowRank()
        @test RowRankNess(UnitLowerTriangular(A)) === IsFullRowRank()
        @test RowRankNess(UnitUpperTriangular(A)) === IsFullRowRank()

        @test traitsof(A) === ()
        @test traitsof(Diagonal(randn(3))) === (IsHermitian(), IsSymmetric(), IsDiagonal())
    end

    @timed_testset "BatchedMulStyle" begin
        @test BatchedMulStyle(x -> 2 * x) === ColumnwiseMul()
        @test BatchedMulStyle(sum) === ColumnwiseMul()
        op = asoperator(randn(3, 3))
        @test BatchedMulStyle(op) === BatchedMul()
        @test BatchedMulStyle(op ∘ op) === BatchedMul()
        @test BatchedMulStyle(op ∘ (x -> 2 * x)) === ColumnwiseMul()
        @test BatchedMulStyle((x -> 2 * x) ∘ op) === ColumnwiseMul()
    end

    @timed_testset "_ispossemidef" begin
        @test MatrixShapedOperators._ispossemidef(Diagonal([0.0, 2.0]))
        @test !MatrixShapedOperators._ispossemidef(Diagonal([-1.0, 2.0]))
        @test MatrixShapedOperators._ispossemidef(2.0 * I)
        @test !MatrixShapedOperators._ispossemidef(-2.0 * I)
        # complex storage with real values is judged by value, non-real
        # values are never psd:
        @test MatrixShapedOperators._ispossemidef(Diagonal(complex.([0.0, 2.0])))
        @test !MatrixShapedOperators._ispossemidef(Diagonal([-1.0 + 0im, 2.0 + 0im]))
        @test !MatrixShapedOperators._ispossemidef(Diagonal([1.0 + 0.5im, 2.0 + 0im]))
        @test MatrixShapedOperators._ispossemidef((2.0 + 0im) * I)
        @test !MatrixShapedOperators._ispossemidef((2.0 + 1im) * I)
        B = randn(3, 3)
        @test MatrixShapedOperators._ispossemidef(B' * B + I)
        @test !MatrixShapedOperators._ispossemidef(-(B' * B) - I)
    end
end
