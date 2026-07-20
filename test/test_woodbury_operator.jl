# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).

using MatrixShapedOperators
using Test
using TimerOutputs: @timed_testset

using LinearAlgebra

using MatrixShapedOperators: Symmetricity, IsSymmetric, UnknownSymmetricity,
    Hermitianity, IsHermitian, UnknownHermitianity,
    PosDefNess, IsPosDef, IsPosSemiDefOnly, UnknownPosDefNess

@timed_testset "woodbury_operator" begin
    n, k = 6, 2
    a = rand(n) .+ 1
    B = randn(n, k)
    D = Symmetric(randn(k, k))
    ref = Diagonal(a) + B * D * B'
    x = randn(n)
    X = randn(n, 3)

    op = WoodburyOperator(Diagonal(a), B, D)
    @test op isa WoodburyOperator{Float64}
    # woodbury_operator is the preferred construction seam:
    @test woodbury_operator(Diagonal(a), B, D) == op
    @test size(op) == (n, n)
    @test size(op, 1) == n
    @test op * x ≈ ref * x
    @test op * x ≈ ref * x
    @test op * X ≈ ref * X
    @test Matrix(op) ≈ ref
    @test op == WoodburyOperator(Diagonal(a), B, D)
    @test occursin("woodbury_operator", sprint(show, op))

    @test eltype(WoodburyOperator(Diagonal(Float32.(a)), Float32.(B), Symmetric(randn(Float32, k, k)))) == Float32
    @test eltype(WoodburyOperator(Diagonal(Float32.(a)), B, D)) == Float64

    @timed_testset "constructor checks" begin
        @test_throws DimensionMismatch WoodburyOperator(randn(3, 4), B, D)
        @test_throws DimensionMismatch WoodburyOperator(Diagonal(a), randn(n + 1, k), D)
        @test_throws DimensionMismatch WoodburyOperator(Diagonal(a), B, Symmetric(randn(k + 1, k + 1)))
        @test_throws ArgumentError WoodburyOperator(Diagonal(a), B, randn(k, k))

        # symmetric-by-trait operator middles are accepted (real
        # symmetric cross-derives hermitian):
        wsym = woodbury_operator(Diagonal(a), B, asoperator(D))
        @test wsym * x ≈ ref * x
    end

    @timed_testset "hash and isapprox" begin
        wp2 = woodbury_operator(Diagonal(copy(a)), copy(B), Symmetric(copy(Matrix(D))))
        wp3 = woodbury_operator(Diagonal(a), B, Symmetric(Matrix(D)))
        @test wp2 == wp3
        @test hash(wp2) == hash(wp3)
        @test wp2 ≈ wp3
    end

    @timed_testset "adjoint" begin
        A2 = randn(n, n)
        wa = WoodburyOperator(A2, B, D)
        @test Matrix(wa') ≈ (A2 + B * D * B')'
        @test wa'.B === B && wa'.D === D
    end

    @timed_testset "traits" begin
        # structural traits delegate to the base (D is symmetric resp.
        # hermitian by construction contract):
        base_pd = asoperator(Diagonal(a), IsPosDef())
        D_psd = asoperator(Diagonal(rand(k)), IsPosSemiDefOnly())
        wpd = WoodburyOperator(base_pd, B, D_psd)
        @test PosDefNess(wpd) === IsPosDef()
        @test Hermitianity(wpd) === IsHermitian()
        @test Symmetricity(wpd) === IsSymmetric()
        @test isposdef(wpd) && ishermitian(wpd) && issymmetric(wpd)

        @test PosDefNess(WoodburyOperator(D_psd, randn(k, k), D_psd)) === IsPosSemiDefOnly()
        @test PosDefNess(op) === UnknownPosDefNess()

        # value-refined is* checks work on plain components:
        @test ishermitian(op) && issymmetric(op)
        @test isposdef(WoodburyOperator(Diagonal(a), B, Symmetric(Matrix(I(k) * 1.0))))
        @test !isposdef(op) || isposdef(Hermitian(Matrix(op)))
    end

    @timed_testset "rowgram_factor" begin
        # tall low-rank update (k < n):
        Dpsd = Symmetric(Matrix(Diagonal([2.0, 0.5])))
        wp = WoodburyOperator(Diagonal(a), B, Dpsd)
        F = rowgram_factor(wp)
        @test F isa MatrixShapedProduct
        @test Matrix(F * F') ≈ Matrix(wp)
        lad = logabsdet(F)
        @test 2 * lad[1] ≈ logdet(Hermitian(Matrix(wp)))

        # indefinite D with a deterministically positive-definite total
        # (the negative update is small against the base):
        aind = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
        Bind = [1.0 0.5; 0.5 1.0; 1.0 -0.5; -0.5 1.0; 0.5 0.5; 1.0 1.0]
        Dind = Symmetric([0.5 0.0; 0.0 -0.25])
        wi = WoodburyOperator(Diagonal(aind), Bind, Dind)
        @test isposdef(Hermitian(Matrix(wi)))
        Fi = rowgram_factor(wi)
        @test Matrix(Fi * Fi') ≈ Matrix(wi)
        @test 2 * first(logabsdet(Fi)) ≈ logdet(Hermitian(Matrix(wi)))

        # singular D (rank-deficient update; the factorization never
        # forms inv(D)):
        Dsing = Symmetric([1.0 0.0; 0.0 0.0])
        ws = WoodburyOperator(Diagonal(aind), Bind, Dsing)
        Fs = rowgram_factor(ws)
        @test Matrix(Fs * Fs') ≈ Matrix(ws)

        # rank-deficient B:
        Brd = [1.0 2.0; 0.5 1.0; 1.0 2.0; -0.5 -1.0; 0.5 1.0; 1.0 2.0]
        wrd = WoodburyOperator(Diagonal(aind), Brd, Dpsd)
        Frd = rowgram_factor(wrd)
        @test Matrix(Frd * Frd') ≈ Matrix(wrd)

        # wide update (k > n):
        nw = 3
        Bw = randn(nw, 5)
        Dw = Symmetric(Matrix(Diagonal(rand(5) .+ 1)))
        ww = WoodburyOperator(Diagonal(rand(nw) .+ 1), Bw, Dw)
        Fw = rowgram_factor(ww)
        @test Matrix(Fw * Fw') ≈ Matrix(ww)
        @test 2 * first(logabsdet(Fw)) ≈ logdet(Hermitian(Matrix(ww)))

        # empty update:
        B0 = zeros(4, 0)
        w0 = WoodburyOperator(Diagonal([1.0, 2.0, 3.0, 4.0]), B0, Symmetric(zeros(0, 0)))
        F0 = rowgram_factor(w0)
        @test Matrix(F0 * F0') ≈ Matrix(w0)

        # operator components with matrix representations:
        wo = WoodburyOperator(asoperator(Diagonal(a)), B, asoperator(Matrix(Dpsd), IsHermitian()))
        Fo = rowgram_factor(wo)
        @test Matrix(Fo * Fo') ≈ Matrix(wo)

        # not positive semi-definite:
        wneg = WoodburyOperator(Diagonal(fill(1e-3, n)), B, Symmetric(Matrix(-10.0 * I(k))))
        @test_throws PosDefException rowgram_factor(wneg)

        @timed_testset "complex" begin
            Bz = randn(ComplexF64, n, k)
            Dz = Hermitian(Matrix(Diagonal([1.5, 0.5])))
            wz = WoodburyOperator(Diagonal(a), Bz, Dz)
            Fz = rowgram_factor(wz)
            @test Matrix(Fz * Fz') ≈ Matrix(wz)
            @test 2 * first(logabsdet(Fz)) ≈ logdet(Hermitian(Matrix(wz)))
            zv = randn(ComplexF64, n)
            @test wz \ zv ≈ Matrix(wz) \ zv
        end
    end

    @timed_testset "inverse application" begin
        Dpsd = Symmetric(Matrix(Diagonal([2.0, 0.5])))
        wp = WoodburyOperator(Diagonal(a), B, Dpsd)
        y = randn(n)
        Y = randn(n, 3)
        @test wp \ y ≈ Matrix(wp) \ y
        @test wp \ Y ≈ Matrix(wp) \ Y

        # solving with a once-obtained factor matches:
        F = rowgram_factor(wp)
        @test F' \ (F \ y) ≈ wp \ y

        # empty update:
        w0 = WoodburyOperator(Diagonal([1.0, 2.0, 3.0, 4.0]), zeros(4, 0), Symmetric(zeros(0, 0)))
        @test w0 \ ones(4) ≈ [1.0, 0.5, 1 / 3, 0.25]

        # the push-through form solves indefinite and singular-D
        # operators, only nonsingularity of the total is required:
        wneg = WoodburyOperator(Diagonal(fill(1e-3, n)), B, Symmetric(Matrix(-10.0 * I(k))))
        @test wneg \ y ≈ Matrix(wneg) \ y
        wsng = WoodburyOperator(Diagonal(a), B, Symmetric([1.0 0.0; 0.0 0.0]))
        @test wsng \ y ≈ Matrix(wsng) \ y
    end

    @timed_testset "logabsdet and factorize" begin
        Dpsd = Symmetric(Matrix(Diagonal([2.0, 0.5])))
        wp = WoodburyOperator(Diagonal(a), B, Dpsd)
        y = randn(n)

        lad = logabsdet(wp)
        lad_ref = logabsdet(Matrix(wp))
        @test lad[1] ≈ lad_ref[1] && lad[2] ≈ lad_ref[2]

        # indefinite total with negative determinant:
        wneg = WoodburyOperator(Diagonal(fill(1e-3, n)), B, Symmetric(Matrix(-10.0 * I(k))))
        ladn = logabsdet(wneg)
        ladn_ref = logabsdet(Matrix(wneg))
        @test ladn[1] ≈ ladn_ref[1] && ladn[2] ≈ ladn_ref[2]

        # empty update:
        w0 = WoodburyOperator(Diagonal([1.0, 2.0, 3.0, 4.0]), zeros(4, 0), Symmetric(zeros(0, 0)))
        @test first(logabsdet(w0)) ≈ log(24.0)

        # empty ambient dimension:
        w00 = WoodburyOperator(zeros(0, 0), zeros(0, 2), Symmetric(Matrix(1.0 * I(2))))
        lad00 = logabsdet(w00)
        @test lad00[1] == 0.0 && lad00[2] == 1.0

        # a singular total yields (-Inf, 0) like for matrices, solving
        # throws:
        wsng2 = WoodburyOperator(Matrix(1.0 * I(2)), reshape([1.0, 0.0], 2, 1), Symmetric(fill(-1.0, 1, 1)))
        ladsng = logabsdet(wsng2)
        @test ladsng[1] == -Inf && ladsng[2] == 0.0
        @test_throws SingularException wsng2 \ [1.0, 2.0]

        # factorize caches the Gram factor for repeated use:
        FW = factorize(wp)
        @test FW isa WoodburyFactorization
        @test size(FW) == (n, n)
        @test FW' === FW
        @test FW \ y ≈ Matrix(wp) \ y
        @test rowgram_factor(FW) isa MatrixShapedOperator
        @test Matrix(rowgram_factor(FW) * rowgram_factor(FW)') ≈ Matrix(wp)
        ladf = logabsdet(FW)
        @test ladf[1] ≈ lad_ref[1] && ladf[2] ≈ lad_ref[2]
        @test FW == factorize(wp)
        @test hash(FW) == hash(factorize(wp))
        @test issuccess(FW)
        @test occursin("WoodburyFactorization", sprint(show, FW))

        # a complex right-hand side on a real factorization exercises
        # the disambiguation against the generic Factorization solve:
        @test FW \ complex.(y) ≈ Matrix(wp) \ complex.(y)

        # Float32 and complex positive-definite factorizations:
        w32 = WoodburyOperator(Diagonal(Float32.(a)), Float32.(B), Symmetric(Matrix(Diagonal(Float32[2, 0.5]))))
        y32 = randn(Float32, n)
        @test factorize(w32) \ y32 ≈ Matrix(w32) \ y32
        Bzp = randn(ComplexF64, n, k)
        wzp = WoodburyOperator(Diagonal(a), Bzp, Hermitian(Matrix(Diagonal([1.5, 0.5]))))
        zp = randn(ComplexF64, n)
        @test factorize(wzp) \ zp ≈ Matrix(wzp) \ zp
    end

    @timed_testset "complex element types" begin
        Bz = randn(ComplexF64, n, k)
        Dz = Hermitian(randn(ComplexF64, k, k))
        Az = Diagonal(rand(n) .+ 1)
        wz = WoodburyOperator(Az, Bz, Dz)
        refz = Az + Bz * Dz * Bz'
        z = randn(ComplexF64, n)
        @test wz * z ≈ refz * z
        @test Matrix(wz') ≈ refz'
        @test Symmetricity(wz) === UnknownSymmetricity()
        @test ishermitian(wz)
        @test_throws ArgumentError WoodburyOperator(Az, Bz, randn(ComplexF64, k, k))

        # complex Diagonal middle matrix with real entries:
        @test isposdef(WoodburyOperator(Az, Bz, Diagonal(complex.([1.5, 0.5]))))
        @test !isposdef(WoodburyOperator(Az, Bz, Diagonal(complex.([1.5, -0.5]))))
    end

    @timed_testset "declared-posdef dense base" begin
        # rowgram_factor and factorize use the operator path, so a
        # declared-posdef base with rounding-level asymmetry works like
        # it does for solves and logabsdet:
        S = let A = randn(4, 4); A * A' + I end
        S[2, 1] = nextfloat(S[2, 1])
        wd = woodbury_operator(
            asoperator(S, MatrixShapedOperators.IsPosDef()),
            randn(4, 2), Symmetric(Matrix(2.0I(2)))
        )
        F = rowgram_factor(wd)
        @test Matrix(F) * Matrix(F)' ≈ Matrix(wd)
        FW = factorize(wd)
        y = randn(4)
        @test FW \ y ≈ Matrix(wd) \ y
    end
end
