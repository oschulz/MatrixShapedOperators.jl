# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).

using MatrixShapedOperators
using Test
using TimerOutputs: @timed_testset

using LinearAlgebra

using MatrixShapedOperators: Symmetricity, IsSymmetric, Hermitianity, IsHermitian,
    PosDefNess, IsPosDef, UnknownPosDefNess,
    Triangularity, IsDiagonal, IsLowerTriangular, IsUpperTriangular,
    RowRankNess, IsFullRowRank, traitsof

@timed_testset "matrix_as_operator" begin
    A = randn(Float32, 4, 5)
    x = randn(Float32, 5)
    X = randn(Float32, 5, 3)

    w = @inferred(asoperator(A))
    @test w isa MatrixAsOperator{Float32,Tuple{},Matrix{Float32}}
    @test MatrixAsOperator(A) == w
    @test MatrixAsOperator(w) === w
    @test asoperator(w) === w
    @test parent(w) === A
    @test asmatrix(w) === A
    @test size(w) == (4, 5)
    @test traitsof(w) === ()

    @test @inferred(w * x) ≈ A * x
    @test w * x ≈ A * x
    @test w * X ≈ A * X
    @test parent(w') == A'
    @test w'' == w

    @test AbstractMatrix(w) == A
    @test AbstractMatrix(w) !== A
    @test Matrix(w) == A

    @test w == asoperator(copy(A))
    @test w != asoperator(randn(Float32, 4, 5))
    @test occursin("MatrixAsOperator", sprint(show, w))

    @timed_testset "value equality and approx" begin
        @test w == A
        @test A == w
        @test w != A .+ 1
        @test isequal(w, A)
        @test hash(w) == hash(A)
        @test w ≈ A
        @test A ≈ w
        @test w ≈ asoperator(A .+ 1.0f-6) rtol = 1.0f-3

        # trait declarations do not participate in equality:
        Ssq = randn(Float32, 4, 4)
        @test asoperator(Ssq * Ssq' + I, IsPosDef()) == asoperator(Ssq * Ssq' + I)

        # value equality across wrapped matrix types:
        d = asoperator(Diagonal([1.0, 2.0]))
        @test d == Diagonal([1.0, 2.0])
        @test d == asoperator(Matrix(Diagonal([1.0, 2.0])))
    end

    @timed_testset "structural traits from the wrapped matrix" begin
        d = asoperator(Diagonal(randn(4)))
        @test traitsof(d) === (IsHermitian(), IsSymmetric(), IsDiagonal())
        @test Triangularity(d) === IsDiagonal()
        @test issymmetric(d) && ishermitian(d)
        @test !isposdef(d)

        l = asoperator(LowerTriangular(randn(3, 3)))
        @test Triangularity(l) === IsLowerTriangular()
        @test Triangularity(l') === IsUpperTriangular()

        u = asoperator(UnitLowerTriangular(randn(3, 3)))
        @test RowRankNess(u) === IsFullRowRank()
    end

    @timed_testset "declared traits" begin
        B = randn(Float32, 5, 5)
        Σ = B' * B + I
        Σop = asoperator(Σ, IsPosDef())
        @test PosDefNess(Σop) === IsPosDef()
        @test Hermitianity(Σop) === IsHermitian()
        @test RowRankNess(Σop) === IsFullRowRank()
        @test isposdef(Σop) && ishermitian(Σop)
        @test occursin("IsPosDef", sprint(show, Σop))

        # declarations merge with the structural traits:
        sd = asoperator(Symmetric(Σ), IsPosDef())
        @test Symmetricity(sd) === IsSymmetric()
        @test PosDefNess(sd) === IsPosDef()

        # adjoint preserves the stored traits via the trait algebra:
        @test PosDefNess(Σop') === IsPosDef()
    end

    @timed_testset "lower_cholesky delegation" begin
        dop = asoperator(Diagonal([4.0, 9.0]))
        lc = lower_cholesky(dop)
        @test lc isa MatrixAsOperator
        @test parent(lc) == Diagonal([2.0, 3.0])

        # known hermitianity skips the value-level check, tolerating
        # rounding-level asymmetries in computed covariances:
        B = randn(4, 4)
        Σ = B * B' + I
        Σ[1, 2] = nextfloat(Σ[1, 2])
        @test !ishermitian(Σ)
        Lσ = lower_cholesky(asoperator(Σ, IsPosDef()))
        @test Matrix(Lσ) * Matrix(Lσ)' ≈ Σ
        @test_throws Exception lower_cholesky(asoperator(Σ))
    end

    @timed_testset "inverse application" begin
        S = randn(4, 4) + 5 * I
        ws = asoperator(S)
        y = randn(4)
        Y = randn(4, 2)
        @test ws \ y ≈ S \ y
        @test ws \ Y ≈ S \ Y

        d5 = asoperator(Diagonal([2.0, 4.0]))
        @test d5 \ [2.0, 8.0] ≈ [1.0, 2.0]

        # known positive definiteness solves via Cholesky, tolerating
        # rounding-level asymmetries:
        B5 = randn(4, 4)
        Σ5 = B5 * B5' + I
        Σ5[1, 2] = nextfloat(Σ5[1, 2])
        @test asoperator(Σ5, IsPosDef()) \ y ≈ Hermitian(Σ5) \ y
    end

    @timed_testset "integer element types" begin
        wi = asoperator([1 2; 3 4])
        @test wi * [1, 1] == [3, 7]
        
        @test Matrix(wi) == [1 2; 3 4]
        @test eltype((wi * zeros(Int, 2, 0))) == Int
    end

    @timed_testset "complex element types" begin
        C = randn(ComplexF64, 3, 3)
        wc = asoperator(C)
        @test eltype(wc) == ComplexF64
        z = randn(ComplexF64, 3)
        @test wc * z ≈ C * z
        @test wc' * z ≈ C' * z
    end

    @timed_testset "AbstractQ" begin
        M = randn(5, 3)
        Q = qr(M).Q
        wq = @inferred(asoperator(Q))
        @test wq isa MatrixAsOperator{Float64}
        @test parent(wq) === Q
        @test size(wq) == (5, 5)
        @test MatrixShapedOperators.Unitarity(wq) === MatrixShapedOperators.IsUnitary()
        @test MatrixShapedOperators.RowRankNess(wq) === MatrixShapedOperators.IsFullRowRank()

        z = randn(5)
        Z = randn(5, 2)
        Qm = Q * Matrix{Float64}(I, 5, 5)
        @test wq * z ≈ Qm * z
        @test wq * Z ≈ Qm * Z
        @test wq' * z ≈ Qm' * z
        @test AbstractMatrix(wq) ≈ Qm
        @test Matrix(wq) ≈ Qm
        @test_throws MethodError asmatrix(wq)

        lad = logabsdet(wq)
        @test lad[1] ≈ 0 atol = 1e-12
        @test lad[2] ≈ det(Qm)

        # Q solves are adjoint applications:
        @test wq \ z ≈ Qm' * z
        @test wq \ Z ≈ Qm' * Z

        # Q operands in the operator algebra:
        a = asoperator(randn(4, 5))
        @test Matrix(a * Q) ≈ Matrix(a) * Qm
        b = asoperator(randn(5, 3))
        @test Matrix(Q * b) ≈ Qm * Matrix(b)
        s = asoperator(randn(5, 5))
        @test Matrix(s + Q) ≈ Matrix(s) + Qm
        @test Matrix(Q + s) ≈ Qm + Matrix(s)

        @timed_testset "logabsdet delegation" begin
            d = asoperator(Diagonal([2.0, -3.0]))
            lad = logabsdet(d)
            @test lad[1] ≈ log(6.0) && lad[2] == -1.0
        end
    end

    @timed_testset "declarations and Q adjoints" begin
        # square-only declarations are rejected on non-square matrices:
        @test_throws ArgumentError asoperator(randn(2, 3), MatrixShapedOperators.IsHermitian())
        @test asoperator(randn(2, 3), MatrixShapedOperators.IsFullRowRank()) isa MatrixAsOperator

        # declarations can be added to an already-wrapped matrix:
        Σd = let A = randn(4, 4); A * A' + I end
        wrapped = asoperator(Σd)
        declared = asoperator(wrapped, MatrixShapedOperators.IsPosDef())
        @test parent(declared) === Σd
        @test isposdef(declared)
        @test asoperator(wrapped) === wrapped

        # declared hermitianity absorbs rounding-level asymmetry, also
        # through a complex Symmetric wrapper:
        Cn = let C = randn(ComplexF64, 3, 3); Matrix(Hermitian(C * C' + I)) end
        Cn[2, 1] += 1e-15im
        Lh = lower_cholesky(asoperator(Symmetric(Cn), MatrixShapedOperators.IsHermitian()))
        @test Matrix(Lh) * Matrix(Lh)' ≈ Matrix(Hermitian(Cn))

        # logabsdet of an adjoint Q:
        Q = qr(randn(4, 4)).Q
        lad = logabsdet(asoperator(Q)')
        @test abs(lad[2]) ≈ 1 && abs(lad[1]) < 1e-10

        # factorize caches the parent factorization, declared positive
        # definiteness selects a Cholesky; structured parents
        # self-return:
        yd = randn(4)
        Fd = factorize(declared)
        @test Fd isa Cholesky
        @test Fd \ yd ≈ Σd \ yd
        Fl = factorize(asoperator(Σd + 5 * ones(4, 4)))
        @test Fl isa LinearAlgebra.Factorization
        @test Fl \ yd ≈ (Σd + 5 * ones(4, 4)) \ yd
        @test factorize(asoperator(Diagonal([2.0, 3.0]))) isa Diagonal
    end
end
