# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).


"""
    woodbury_operator(A::MatrixShaped, B::MatrixShaped, D::MatrixShaped)

Returns a matrix-shaped operator `op` that represents `A + B * D * B'` for a
square base `A`, an `n × k` update factor `B` (typically tall, `k`
small) and a square symmetric
(hermitian for complex element types) middle matrix `D`.

Typically returns a [`WoodburyOperator`](@ref):

```julia
op = woodbury_operator(A, B, D)
op * x == A * x + B * (D * (B' * x))
```

A [`𝟘`](@ref) base yields a pure low-rank `B * D * B'`, which supports
application and traits but no solves or determinants - the push-through
formulas require an invertible base.

The operator is symmetric/hermitian if `A` is, and
positive (semi-)definite if `A` is and `D` is positive semi-definite.
Components are [`MatrixShaped`](@ref); keep them structured (`Diagonal`
base, triangular `D`) to enable structural operations.
"""
woodbury_operator(A, B, D) = WoodburyOperator(A, B, D)
export woodbury_operator


"""
    struct WoodburyOperator{T<:Number,TA,TB,TD} <: MatrixShapedOperator{T}

Represents `A + B * D * B'`.

User code should not call `WoodburyOperator` directly, but use
[`woodbury_operator`](@ref) instead.

Application costs one application per component, O(n k) for a
structured base.
"""
struct WoodburyOperator{T<:Number,TA,TB,TD} <: MatrixShapedOperator{T}
    A::TA
    B::TB
    D::TD
end
export WoodburyOperator

function WoodburyOperator(A::TA, B::TB, D::TD) where {TA,TB,TD}
    size(A, 1) == size(A, 2) || throw(DimensionMismatch(
        "WoodburyOperator base must be square, got size $(size(A))"
    ))
    size(B, 1) == size(A, 1) || throw(DimensionMismatch(
        "WoodburyOperator low-rank factor of size $(size(B)) does not match base of size $(size(A))"
    ))
    size(D, 1) == size(D, 2) == size(B, 2) || throw(DimensionMismatch(
        "WoodburyOperator middle matrix of size $(size(D)) does not match factor of size $(size(B))"
    ))
    _is_hermitian_component(D) || throw(ArgumentError(
        "WoodburyOperator middle matrix must be symmetric resp. hermitian"
    ))
    T = promote_type(eltype(A), eltype(B), eltype(D))
    return WoodburyOperator{T,TA,TB,TD}(A, B, D)
end

# The is* query cross-derives real symmetric => hermitian and stays
# trait-based (traceable) on operators:
_is_hermitian_component(D) = ishermitian(D)

Base.size(op::WoodburyOperator) = size(op.A)

Base.adjoint(op::WoodburyOperator) = WoodburyOperator(adjoint(op.A), op.B, op.D)

# D is symmetric resp. hermitian by construction contract:
Symmetricity(op::WoodburyOperator{T}) where T =
    NumberDomain(T) === Real ? Symmetricity(op.A) : UnknownSymmetricity()
Hermitianity(op::WoodburyOperator) = Hermitianity(op.A)

PosDefNess(op::WoodburyOperator) = _woodbury_posdefness(PosDefNess(op.A), PosDefNess(op.D))
_woodbury_posdefness(::IsPosDef, ::IsPosSemiDef) = IsPosDef()
_woodbury_posdefness(::IsPosSemiDefOnly, ::IsPosSemiDef) = IsPosSemiDefOnly()
_woodbury_posdefness(::PosDefNess, ::PosDefNess) = UnknownPosDefNess()

LinearAlgebra.issymmetric(op::WoodburyOperator{T}) where T = NumberDomain(T) === Real && issymmetric(op.A)
LinearAlgebra.ishermitian(op::WoodburyOperator) = ishermitian(op.A)
LinearAlgebra.isposdef(op::WoodburyOperator) = isposdef(op.A) && _ispossemidef(op.D)

explicit_mul_impl(op::WoodburyOperator, v::AbstractVector{<:Number}) =
    _apply(op.A, v) .+ _apply(op.B, _apply(op.D, _apply(adjoint(op.B), v)))
batched_mul_impl(op::WoodburyOperator, X::AbstractMatrix{<:Number}) =
    _apply(op.A, X) .+ _apply(op.B, _apply(op.D, _apply(adjoint(op.B), X)))

"""
    rowgram_factor(op::WoodburyOperator)

Computes a factor `F` with `op == F * F'` via the numerically stable
factorization `L_A * Q * blockdiag(V', I)` with `Q R = L_A \\ B` and
`V' V = I + R D R'` (Zhang et al. 2022, appendix A), in O(n k²).

# Extended help

The result is a lazy operator product with structural `logabsdet`.
Requires a positive-definite base with a structural
[`lower_cholesky`](@ref) and a positive-definite total `A + B D B'`
(`D` itself may be indefinite or singular; throws `PosDefException`
otherwise), with update components `B` and `D` that have explicit
matrix representations ([`asmatrix`](@ref)).
"""
function rowgram_factor(op::WoodburyOperator{T}) where T
    # The base factorizes on the operator path, so declared posdef
    # structure applies, like for `\` and `logabsdet`:
    Aop = asoperator(op.A)
    B = asmatrix(op.B)
    D = asmatrix(op.D)
    Lop = lower_cholesky(Aop)
    # Not an optimization but a correctness guard: det/logabsdet of the
    # Q of a zero-column qr throws, and rank-zero updates do occur
    # (e.g. L-BFGS representations with empty history):
    size(B, 2) == 0 && return Lop

    F = qr(Lop \ B)
    R = F.R
    Vt = lower_cholesky(Hermitian(muladd(R, D * R', I)))
    n = size(Aop, 1)
    m = size(R, 1)

    Qop = asoperator(F.Q)
    Vtop = asoperator(Vt)
    return m == n ? Lop * Qop * Vtop :
        Lop * Qop * blockdiag_operator(Vtop, one(T) * IdentityOperator(n - m))
end

# Structural solve via the D-inverse-free push-through form
# (A + B D B')⁻¹ x = A⁻¹x - A⁻¹B (I + D B'A⁻¹B)⁻¹ D B'A⁻¹x. Requires
# an invertible base with a structural solve and a nonsingular total;
# D may be indefinite or singular. Each solve refactorizes, use
# `LinearAlgebra.factorize` for repeated solves of positive-definite
# operators. The reduced system solves via function-level `\` (not an
# explicit lu) so that AD rules for `\` apply:
function ldiv_impl(op::WoodburyOperator, x::AbstractVecOrMat{<:Number})
    Aop = asoperator(op.A)
    size(op.B, 2) == 0 && return ldiv_impl(Aop, x)
    B = asmatrix(op.B)
    D = asmatrix(op.D)
    Ainv_x = ldiv_impl(Aop, x)
    Ainv_B = ldiv_impl(Aop, B)
    S = muladd(D, adjoint(B) * Ainv_B, I)
    return Ainv_x .- Ainv_B * (S \ (D * (adjoint(B) * Ainv_x)))
end

# det(A + B D B') = det(A) det(I + D B'A⁻¹B), the matching D-inverse-
# free push-through form; requires a structural solve and logabsdet of
# an invertible base. A singular total yields (-Inf, 0), like for
# matrices (function-level `logabsdet` factorizes with check = false
# itself; it also keeps AD rules for `logabsdet` applicable):
function LinearAlgebra.logabsdet(op::WoodburyOperator)
    Aop = asoperator(op.A)
    la_A = logabsdet(Aop)
    size(op.B, 2) == 0 && return la_A
    B = asmatrix(op.B)
    D = asmatrix(op.D)
    la_S = logabsdet(muladd(D, adjoint(B) * ldiv_impl(Aop, B), I))
    return first(la_A) + first(la_S), last(la_A) * last(la_S)
end

Base.:(==)(a::WoodburyOperator, b::WoodburyOperator) =
    a.A == b.A && a.B == b.B && a.D == b.D

Base.hash(op::WoodburyOperator, h::UInt) = hash(op.D, hash(op.B, hash(op.A, hash(:WoodburyOperator, h))))

Base.isapprox(a::WoodburyOperator, b::WoodburyOperator; kwargs...) =
    isapprox(a.A, b.A; kwargs...) && isapprox(a.B, b.B; kwargs...) && isapprox(a.D, b.D; kwargs...)

function Base.show(io::IO, op::WoodburyOperator)
    print(io, "woodbury_operator(")
    show(io, op.A)
    print(io, ", ")
    show(io, op.B)
    print(io, ", ")
    show(io, op.D)
    print(io, ")")
end


"""
    struct WoodburyFactorization{T<:Number} <: LinearAlgebra.Factorization{T}

Cached Gram factorization `op == F * F'` of a positive-definite
[`WoodburyOperator`](@ref).

User code should not call `WoodburyFactorization` directly, but use
[`LinearAlgebra.factorize`](@ref) instead. The result supports `\\`,
`logabsdet` and [`rowgram_factor`](@ref) without refactorizing:

```julia
op = woodbury_operator(A, B, D)
FW = factorize(op)
FW \\ b
logabsdet(FW)
F = rowgram_factor(FW)  # e.g. for sampling and whitening
```
"""
struct WoodburyFactorization{T<:Number,F<:MatrixShapedOperator{T}} <: LinearAlgebra.Factorization{T}
    F::F
end
export WoodburyFactorization

"""
    LinearAlgebra.factorize(op::WoodburyOperator)

Returns a [`WoodburyFactorization`](@ref) holding the
[`rowgram_factor`](@ref) of `op`, computed once, for repeated solves,
determinants and factor access. Same preconditions as
`rowgram_factor`.
"""
LinearAlgebra.factorize(op::WoodburyOperator) = WoodburyFactorization(rowgram_factor(op))

Base.size(fac::WoodburyFactorization) = size(fac.F)
Base.size(fac::WoodburyFactorization, i::Integer) = size(fac.F, i)

# The factorized operator is hermitian:
Base.adjoint(fac::WoodburyFactorization) = fac

Base.:(\)(fac::WoodburyFactorization, x::AbstractVecOrMat{<:Number}) = _wfac_solve(fac, x)

# Disambiguation against the generic complex-RHS Factorization solve:
Base.:(\)(fac::WoodburyFactorization{T}, x::Union{Vector{Complex{T}},Matrix{Complex{T}}}) where {T<:Union{Float32,Float64}} =
    _wfac_solve(fac, x)

function _wfac_solve(fac::WoodburyFactorization, x::AbstractVecOrMat{<:Number})
    F = fac.F
    return adjoint(F) \ (F \ x)
end

# det(F F') == |det(F)|²:
function LinearAlgebra.logabsdet(fac::WoodburyFactorization)
    la = logabsdet(fac.F)
    return 2 * first(la), one(last(la))
end

rowgram_factor(fac::WoodburyFactorization) = fac.F

# Construction throws on failure, an existing factorization is valid:
LinearAlgebra.issuccess(::WoodburyFactorization) = true

Base.:(==)(a::WoodburyFactorization, b::WoodburyFactorization) = a.F == b.F

Base.hash(fac::WoodburyFactorization, h::UInt) = hash(fac.F, hash(:WoodburyFactorization, h))

function Base.show(io::IO, fac::WoodburyFactorization)
    print(io, "WoodburyFactorization(")
    show(io, fac.F)
    print(io, ")")
end


# ToDo:
# * lazy_add_impl collapses: Woodbury + Woodbury with compatible bases
#   (concatenate Bs, block-diagonal Ds), Woodbury + Diagonal/uniform
#   scaling (absorb into base) - keeps means/sums of Woodburys O(n k),
#   but grows the reduced system, so only worthwhile for a demonstrated
#   expression pattern. PDMat interop deliberately stays downstream,
#   PDMat is not tracing-friendly by design.
# * a true triangular lower_cholesky would require rank-k Cholesky
#   updates - a computation, not structural access, so deliberately NOT
#   a lower_cholesky method
