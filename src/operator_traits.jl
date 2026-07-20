# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).

# Trait vocabulary:
#
# * Structure traits, queried via their abstract-type constructors
#   (`Symmetricity(A)`, `PosDefNess(A)`, `Unitarity(A)`, ...): `Is*`
#   results hold for any element values of the representation,
#   `Unknown*` means "not guaranteed". Value-independent, safe to branch
#   on under program tracing, and constant-folding wherever the type
#   alone decides them.
# * `is*`: definitive on explicit matrices (the Base contract, whatever
#   the cost). On operators it defaults to the structure traits - `true`
#   is definitive, `false` means "not derivable" - refined by cheap
#   value checks; do not branch on it in traced code.
# * Capability styles (e.g. `BatchedMulStyle`): select between
#   algorithm variants that are all valid.
#
# Methods for stdlib types live here, methods for operator types with
# the types that implement them.


"""
    MatrixShapedOperators.NumberDomain(A)::Union{Type{Real},Type{Complex}}

Returns `Real` or `Complex`, the number domain of the element type of `A`.

# Extended help

Judged by `real(T) === T` rather than `T <: Real`, so that tracing number
types count as real. Definitive and constant-folding, dispatch
on `::Type{Real}`/`::Type{Complex}` to scope real-only behavior (e.g.
structural symmetry claims, `transpose == adjoint`).
"""
NumberDomain(A) = NumberDomain(eltype(A))
NumberDomain(::Type{T}) where {T<:Number} = real(T) === T ? Real : Complex
@compat public NumberDomain


# Trait manipulation and extraction; each trait category defines its
# trait_category-terminating abstract type, singletons and unknowntrait
# method below.

"""
    MatrixShapedOperators.trait_category(t)::Type

Returns the abstract trait type a trait value belongs to, e.g.
`PosDefNess` for `IsPosDef()`: the last supertype before `Any`.
"""
trait_category(t) = trait_category(typeof(t))
function trait_category(::Type{T}) where T
    S = supertype(T)
    return S === Any ? T : trait_category(S)
end
@compat public trait_category

"""
    MatrixShapedOperators.unknowntrait(::Type{C})

Returns the `Unknown*` value of trait category `C`. Implemented by each
trait category, including third-party ones.
"""
function unknowntrait end
@compat public unknowntrait

"""
    MatrixShapedOperators.traitset(decls...)::Tuple

Returns the canonical trait set for operator types that store their structural
properties as a tuple (type) of traits.

The result is a tuple of the non-`Unknown*` trait values, deduplicated
(the first non-`Unknown*` declaration per category wins, `Unknown*` values
carry no information), sorted by category name. Equal trait sets have equal types.
Implications are closed: unitarity and positive definiteness imply full row
rank, positive semi-definiteness implies hermitian, diagonality implies
symmetric.
"""
traitset(decls...) = _traitset_canonical(_close_implications(_dedupe(_dropunknowns(decls...)))...)
@compat public traitset

# First non-Unknown declaration per category wins; deduplication must
# precede the implication closure so that discarded declarations do not
# contribute implications:
_dedupe(decls::Tuple) = _dedupe_impl((), decls...)
_dedupe_impl(kept::Tuple) = kept
function _dedupe_impl(kept::Tuple, t, rest...)
    return _hastrait(trait_category(t), kept...) ? _dedupe_impl(kept, rest...) : _dedupe_impl((kept..., t), rest...)
end

function _close_implications(decls::Tuple)
    d1 = _hastrait(IsUnitary, decls...) ? (decls..., IsFullRowRank()) : decls
    d2 = _hastrait(IsPosDef, d1...) ? (d1..., IsFullRowRank()) : d1
    d3 = _hastrait(IsPosSemiDef, d2...) ? (d2..., IsHermitian()) : d2
    return _hastrait(IsDiagonal, d3...) ? (d3..., IsSymmetric()) : d3
end

_hastrait(::Type{C}) where C = false
_hastrait(::Type{C}, t, rest...) where C = t isa C || _hastrait(C, rest...)

# Unknown* values carry no information. This runs as ordinary dispatch
# (not inside the generated function): @generated bodies execute in a
# frozen world age and must not call user-extensible functions like
# unknowntrait - they may only emit such calls and use Base reflection.
_dropunknowns() = ()
function _dropunknowns(t, rest...)
    tail = _dropunknowns(rest...)
    return t === unknowntrait(trait_category(t)) ? tail : (t, tail...)
end

# Canonical order is by module and name of the trait category, so that
# equally-named third-party categories cannot collide:
@generated function _traitset_canonical(decls...)
    kept = Type[]
    cats = Type[]
    for T in decls
        C = trait_category(T)
        C in cats && continue
        push!(cats, C)
        push!(kept, T)
    end
    sort!(kept; by = T -> _canonical_key(trait_category(T)))
    return :( ($(map(T -> :($T()), kept)...),) )
end

_canonical_key(C::Type) = (string(parentmodule(C)), String(nameof(C)))

@generated _traits_instance(::Type{TRS}) where {TRS<:Tuple} =
    :( ($(map(T -> :($T()), TRS.parameters)...),) )

"""
    MatrixShapedOperators.gettrait(traits, C::Type)

Returns the trait of category `C` from a [`traitset`](@ref)-created
trait set (a tuple or its type), `unknowntrait(C)` if absent.

Lookup is invariant to the order of the tuple parameters, the canonical
order only serves type identity.
"""
gettrait(traits::Tuple, ::Type{C}) where C = gettrait(typeof(traits), C)
@generated function gettrait(::Type{TRS}, ::Type{C}) where {TRS<:Tuple,C}
    for T in TRS.parameters
        T <: C && return :( $T() )
    end
    return :( unknowntrait(C) )
end
@compat public gettrait

"""
    MatrixShapedOperators.traitsof(A)::Tuple

Returns the canonical trait set of the structure traits of `A`.
"""
traitsof(A) = traitset(
    Symmetricity(A), Hermitianity(A), PosDefNess(A),
    Triangularity(A), Unitarity(A), RowRankNess(A),
)
@compat public traitsof

"""
    MatrixShapedOperators.trait_add(x, y)
    MatrixShapedOperators.trait_mul(x, y)
    MatrixShapedOperators.trait_blockdiag(x, y)
    MatrixShapedOperators.trait_adjoint(x)
    MatrixShapedOperators.trait_scale(x)

Trait algebra: returns the resulting trait of one category under
operator addition, multiplication (composition), block-diagonal
concatenation, adjoint and scaling by a real scalar of unknown value.
Conservative `Unknown*` fallbacks; each trait category defines its
preservation rules in place.
"""
trait_add(x, y) = unknowntrait(trait_category(x))
trait_mul(x, y) = unknowntrait(trait_category(x))
trait_blockdiag(x, y) = unknowntrait(trait_category(x))
trait_adjoint(x) = unknowntrait(trait_category(x))
trait_scale(x) = unknowntrait(trait_category(x))
@compat public trait_add, trait_mul, trait_blockdiag, trait_adjoint, trait_scale

"""
    MatrixShapedOperators.traitset_add(ta::Tuple, tb::Tuple)::Tuple
    MatrixShapedOperators.traitset_mul(ta::Tuple, tb::Tuple)::Tuple
    MatrixShapedOperators.traitset_blockdiag(ta::Tuple, tb::Tuple)::Tuple
    MatrixShapedOperators.traitset_adjoint(ts::Tuple)::Tuple
    MatrixShapedOperators.traitset_scale(ts::Tuple)::Tuple

Trait algebra on whole trait sets: applies [`trait_add`](@ref) etc. per
category and re-canonicalizes the result.
"""
@generated function traitset_add(ta::Tuple, tb::Tuple)
    cats = Type[]
    for T in (ta.parameters..., tb.parameters...)
        C = trait_category(T)
        C in cats || push!(cats, C)
    end
    exs = [:( trait_add(gettrait(ta, $C), gettrait(tb, $C)) ) for C in cats]
    return :( traitset($(exs...)) )
end

@generated function traitset_mul(ta::Tuple, tb::Tuple)
    cats = Type[]
    for T in (ta.parameters..., tb.parameters...)
        C = trait_category(T)
        C in cats || push!(cats, C)
    end
    exs = [:( trait_mul(gettrait(ta, $C), gettrait(tb, $C)) ) for C in cats]
    return :( traitset($(exs...)) )
end

@generated function traitset_blockdiag(ta::Tuple, tb::Tuple)
    cats = Type[]
    for T in (ta.parameters..., tb.parameters...)
        C = trait_category(T)
        C in cats || push!(cats, C)
    end
    exs = [:( trait_blockdiag(gettrait(ta, $C), gettrait(tb, $C)) ) for C in cats]
    return :( traitset($(exs...)) )
end

traitset_adjoint(ts::Tuple) = traitset(map(trait_adjoint, ts)...)
traitset_scale(ts::Tuple) = traitset(map(trait_scale, ts)...)
@compat public traitset_add, traitset_mul, traitset_blockdiag, traitset_adjoint, traitset_scale


"""
    abstract type MatrixShapedOperators.Symmetricity

Structural symmetry trait: `Symmetricity(A)` returns
[`IsSymmetric`](@ref)`()` if `A` is symmetric for any element values
of its representation, [`UnknownSymmetricity`](@ref)`()` otherwise.
"""
abstract type Symmetricity end
@compat public Symmetricity

"""
    MatrixShapedOperators.IsSymmetric <: MatrixShapedOperators.Symmetricity

Symmetric for any element values, see [`Symmetricity`](@ref).
"""
struct IsSymmetric <: Symmetricity end
@compat public IsSymmetric

"""
    MatrixShapedOperators.UnknownSymmetricity <: MatrixShapedOperators.Symmetricity

Symmetry not guaranteed by the representation, see [`Symmetricity`](@ref).
"""
struct UnknownSymmetricity <: Symmetricity end
@compat public UnknownSymmetricity

unknowntrait(::Type{Symmetricity}) = UnknownSymmetricity()

trait_add(::IsSymmetric, ::IsSymmetric) = IsSymmetric()
trait_blockdiag(::IsSymmetric, ::IsSymmetric) = IsSymmetric()
trait_adjoint(x::Symmetricity) = x
trait_scale(x::Symmetricity) = x

Symmetricity(::Any) = UnknownSymmetricity()
Symmetricity(::Symmetric) = IsSymmetric()
Symmetricity(::SymTridiagonal) = IsSymmetric()
Symmetricity(::Diagonal) = IsSymmetric()
Symmetricity(::Hermitian{T}) where {T<:Number} =
    NumberDomain(T) === Real ? IsSymmetric() : UnknownSymmetricity()


"""
    abstract type MatrixShapedOperators.Hermitianity

Structural hermitianity trait: `Hermitianity(A)` returns
[`IsHermitian`](@ref)`()` or [`UnknownHermitianity`](@ref)`()`.
"""
abstract type Hermitianity end
@compat public Hermitianity

"""
    MatrixShapedOperators.IsHermitian <: MatrixShapedOperators.Hermitianity

Hermitian for any element values, see [`Hermitianity`](@ref).
"""
struct IsHermitian <: Hermitianity end
@compat public IsHermitian

"""
    MatrixShapedOperators.UnknownHermitianity <: MatrixShapedOperators.Hermitianity

Hermitianity not guaranteed by the representation, see
[`Hermitianity`](@ref).
"""
struct UnknownHermitianity <: Hermitianity end
@compat public UnknownHermitianity

unknowntrait(::Type{Hermitianity}) = UnknownHermitianity()

trait_add(::IsHermitian, ::IsHermitian) = IsHermitian()
trait_blockdiag(::IsHermitian, ::IsHermitian) = IsHermitian()
trait_adjoint(x::Hermitianity) = x
trait_scale(x::Hermitianity) = x

Hermitianity(::Any) = UnknownHermitianity()
Hermitianity(::Hermitian) = IsHermitian()
Hermitianity(::Diagonal{T}) where {T<:Number} =
    NumberDomain(T) === Real ? IsHermitian() : UnknownHermitianity()
Hermitianity(::Symmetric{T}) where {T<:Number} =
    NumberDomain(T) === Real ? IsHermitian() : UnknownHermitianity()
Hermitianity(::SymTridiagonal{T}) where {T<:Number} =
    NumberDomain(T) === Real ? IsHermitian() : UnknownHermitianity()


"""
    abstract type MatrixShapedOperators.PosDefNess

Structural definiteness trait: `PosDefNess(A)` returns
[`IsPosDef`](@ref)`()`, [`IsPosSemiDefOnly`](@ref)`()` or
[`UnknownPosDefNess`](@ref)`()`. The two `Is*` types subtype the
abstract [`IsPosSemiDef`](@ref) (definite implies semi-definite),
so `PosDefNess(A) isa IsPosSemiDef` tests "guaranteed positive
semi-definite".
"""
abstract type PosDefNess end
@compat public PosDefNess

"""
    MatrixShapedOperators.IsPosSemiDef <: MatrixShapedOperators.PosDefNess

Abstract: positive semi-definite for any element values, definite or
not, see [`PosDefNess`](@ref).
"""
abstract type IsPosSemiDef <: PosDefNess end
@compat public IsPosSemiDef

"""
    MatrixShapedOperators.IsPosDef <: MatrixShapedOperators.IsPosSemiDef

Positive definite for any element values, see [`PosDefNess`](@ref).
"""
struct IsPosDef <: IsPosSemiDef end
@compat public IsPosDef

"""
    MatrixShapedOperators.IsPosSemiDefOnly <: MatrixShapedOperators.IsPosSemiDef

Positive semi-definite for any element values, not known definite, see
[`PosDefNess`](@ref).
"""
struct IsPosSemiDefOnly <: IsPosSemiDef end
@compat public IsPosSemiDefOnly

"""
    MatrixShapedOperators.UnknownPosDefNess <: MatrixShapedOperators.PosDefNess

Definiteness not guaranteed by the representation, see
[`PosDefNess`](@ref).
"""
struct UnknownPosDefNess <: PosDefNess end
@compat public UnknownPosDefNess

unknowntrait(::Type{PosDefNess}) = UnknownPosDefNess()

# psd is preserved under addition, one posdef summand makes it posdef:
trait_add(::IsPosDef, ::IsPosDef) = IsPosDef()
trait_add(::IsPosDef, ::IsPosSemiDef) = IsPosDef()
trait_add(::IsPosSemiDef, ::IsPosDef) = IsPosDef()
trait_add(::IsPosSemiDef, ::IsPosSemiDef) = IsPosSemiDefOnly()
# eigenvalues of a direct sum are the union of the block eigenvalues:
trait_blockdiag(::IsPosDef, ::IsPosDef) = IsPosDef()
trait_blockdiag(::IsPosSemiDef, ::IsPosSemiDef) = IsPosSemiDefOnly()
trait_adjoint(x::PosDefNess) = x

PosDefNess(::Any) = UnknownPosDefNess()


"""
    abstract type MatrixShapedOperators.Triangularity

Structural triangularity trait: `Triangularity(A)` returns
[`IsDiagonal`](@ref)`()`, [`IsLowerTriangular`](@ref)`()`,
[`IsUpperTriangular`](@ref)`()` or [`UnknownTriangularity`](@ref)`()`.
The unions [`IsLowerTriangularOrDiagonal`](@ref) and
[`IsUpperTriangularOrDiagonal`](@ref) match diagonal as well;
`adjoint` swaps lower and upper.
"""
abstract type Triangularity end
@compat public Triangularity

"""
    MatrixShapedOperators.IsDiagonal <: MatrixShapedOperators.Triangularity

Diagonal for any element values, see [`Triangularity`](@ref).
"""
struct IsDiagonal <: Triangularity end
@compat public IsDiagonal

"""
    MatrixShapedOperators.IsLowerTriangular <: MatrixShapedOperators.Triangularity

Lower triangular for any element values, see [`Triangularity`](@ref).
"""
struct IsLowerTriangular <: Triangularity end
@compat public IsLowerTriangular

"""
    MatrixShapedOperators.IsUpperTriangular <: MatrixShapedOperators.Triangularity

Upper triangular for any element values, see [`Triangularity`](@ref).
"""
struct IsUpperTriangular <: Triangularity end
@compat public IsUpperTriangular

"""
    MatrixShapedOperators.UnknownTriangularity <: MatrixShapedOperators.Triangularity

Triangularity not guaranteed by the representation, see
[`Triangularity`](@ref).
"""
struct UnknownTriangularity <: Triangularity end
@compat public UnknownTriangularity

"""
    MatrixShapedOperators.IsLowerTriangularOrDiagonal

Union of [`IsLowerTriangular`](@ref) and [`IsDiagonal`](@ref),
see [`Triangularity`](@ref).
"""
const IsLowerTriangularOrDiagonal = Union{IsLowerTriangular,IsDiagonal}
@compat public IsLowerTriangularOrDiagonal

"""
    MatrixShapedOperators.IsUpperTriangularOrDiagonal

Union of [`IsUpperTriangular`](@ref) and [`IsDiagonal`](@ref),
see [`Triangularity`](@ref).
"""
const IsUpperTriangularOrDiagonal = Union{IsUpperTriangular,IsDiagonal}
@compat public IsUpperTriangularOrDiagonal

unknowntrait(::Type{Triangularity}) = UnknownTriangularity()

# triangularity is preserved under addition and composition, diagonal
# absorbs into either triangle; adjoint swaps lower and upper:
trait_add(::IsDiagonal, ::IsDiagonal) = IsDiagonal()
trait_add(::IsLowerTriangularOrDiagonal, ::IsLowerTriangularOrDiagonal) = IsLowerTriangular()
trait_add(::IsUpperTriangularOrDiagonal, ::IsUpperTriangularOrDiagonal) = IsUpperTriangular()
trait_mul(::IsDiagonal, ::IsDiagonal) = IsDiagonal()
trait_mul(::IsLowerTriangularOrDiagonal, ::IsLowerTriangularOrDiagonal) = IsLowerTriangular()
trait_mul(::IsUpperTriangularOrDiagonal, ::IsUpperTriangularOrDiagonal) = IsUpperTriangular()
trait_adjoint(::IsDiagonal) = IsDiagonal()
trait_adjoint(::IsLowerTriangular) = IsUpperTriangular()
trait_adjoint(::IsUpperTriangular) = IsLowerTriangular()
trait_scale(x::Triangularity) = x

Triangularity(::Any) = UnknownTriangularity()
Triangularity(::LowerTriangular) = IsLowerTriangular()
Triangularity(::UnitLowerTriangular) = IsLowerTriangular()
Triangularity(::UpperTriangular) = IsUpperTriangular()
Triangularity(::UnitUpperTriangular) = IsUpperTriangular()
Triangularity(::Diagonal) = IsDiagonal()
Triangularity(A::Bidiagonal) = A.uplo == 'L' ? IsLowerTriangular() : IsUpperTriangular()


"""
    abstract type MatrixShapedOperators.Unitarity

Structural unitarity trait: `Unitarity(A)` returns
[`IsUnitary`](@ref)`()` if `A` is unitary (orthogonal, for real element
types) for any element values of its representation,
[`UnknownUnitarity`](@ref)`()` otherwise.

Unitary operators are square with full rank, their inverse is their
adjoint and their `logabsdet` is zero.
"""
abstract type Unitarity end
@compat public Unitarity

"""
    MatrixShapedOperators.IsUnitary <: MatrixShapedOperators.Unitarity

Unitary for any element values, see [`Unitarity`](@ref).
"""
struct IsUnitary <: Unitarity end
@compat public IsUnitary

"""
    MatrixShapedOperators.UnknownUnitarity <: MatrixShapedOperators.Unitarity

Unitarity not guaranteed by the representation, see [`Unitarity`](@ref).
"""
struct UnknownUnitarity <: Unitarity end
@compat public UnknownUnitarity

unknowntrait(::Type{Unitarity}) = UnknownUnitarity()

trait_mul(::IsUnitary, ::IsUnitary) = IsUnitary()
trait_blockdiag(::IsUnitary, ::IsUnitary) = IsUnitary()
trait_adjoint(x::Unitarity) = x

Unitarity(::Any) = UnknownUnitarity()
Unitarity(::LinearAlgebra.AbstractQ) = IsUnitary()


"""
    abstract type MatrixShapedOperators.RowRankNess

Structural row-rank trait: `RowRankNess(A)` returns
[`IsFullRowRank`](@ref)`()` or [`UnknownRowRankNess`](@ref)`()`.
Unitarity implies full row rank.
"""
abstract type RowRankNess end
@compat public RowRankNess

"""
    MatrixShapedOperators.IsFullRowRank <: MatrixShapedOperators.RowRankNess

Full row rank for any element values (e.g. unit-triangular types), see
[`RowRankNess`](@ref).
"""
struct IsFullRowRank <: RowRankNess end
@compat public IsFullRowRank

"""
    MatrixShapedOperators.UnknownRowRankNess <: MatrixShapedOperators.RowRankNess

Row rank not guaranteed by the representation, see [`RowRankNess`](@ref).
"""
struct UnknownRowRankNess <: RowRankNess end
@compat public UnknownRowRankNess

unknowntrait(::Type{RowRankNess}) = UnknownRowRankNess()

# surjective maps compose surjectively, direct sums stay surjective:
trait_mul(::IsFullRowRank, ::IsFullRowRank) = IsFullRowRank()
trait_blockdiag(::IsFullRowRank, ::IsFullRowRank) = IsFullRowRank()

RowRankNess(A) = _rowrank_from_unitarity(Unitarity(A))
_rowrank_from_unitarity(::IsUnitary) = IsFullRowRank()
_rowrank_from_unitarity(::Unitarity) = UnknownRowRankNess()

RowRankNess(::UnitLowerTriangular) = IsFullRowRank()
RowRankNess(::UnitUpperTriangular) = IsFullRowRank()


"""
    MatrixShapedOperators.StructureTrait

Union of the structure trait types provided by this package:
[`Symmetricity`](@ref), [`Hermitianity`](@ref), [`PosDefNess`](@ref),
[`Triangularity`](@ref), [`Unitarity`](@ref) and [`RowRankNess`](@ref).

The trait-set machinery ([`traitset`](@ref), [`gettrait`](@ref)) is not
limited to these: a third-party trait needs an abstract type with
singleton subtypes and an [`unknowntrait`](@ref) method.
"""
const StructureTrait = Union{Symmetricity,Hermitianity,PosDefNess,Triangularity,Unitarity,RowRankNess}
@compat public StructureTrait


"""
    abstract type MatrixShapedOperators.BatchedMulStyle

How a multiplication function `f`, as used by [`MulFuncOperator`](@ref)
and [`mulfunc_operator`](@ref), handles matrix arguments:
`BatchedMulStyle(f)` returns [`BatchedMul`](@ref)`()` if `f` applies to
matrix columns as a batch natively (e.g. batched autodiff Jacobian
application, much more efficient especially under program tracing),
[`ColumnwiseMul`](@ref)`()` if `f` must be applied column by column.
"""
abstract type BatchedMulStyle end
@compat public BatchedMulStyle

"""
    MatrixShapedOperators.BatchedMul <: MatrixShapedOperators.BatchedMulStyle

Handles matrix arguments natively as column batches, see
[`BatchedMulStyle`](@ref).
"""
struct BatchedMul <: BatchedMulStyle end
@compat public BatchedMul

"""
    MatrixShapedOperators.ColumnwiseMul <: MatrixShapedOperators.BatchedMulStyle

Applied column by column to matrix arguments, see
[`BatchedMulStyle`](@ref).
"""
struct ColumnwiseMul <: BatchedMulStyle end
@compat public ColumnwiseMul

BatchedMulStyle(::Any) = ColumnwiseMul()

# Matrix-shaped operators, and `Base.Fix1(*, op)` multiplication
# functions built on them, accept matrix arguments natively:
BatchedMulStyle(::MatrixShapedOperator) = BatchedMul()
BatchedMulStyle(::Base.Fix1{typeof(*),<:MatrixShapedOperator}) = BatchedMul()

BatchedMulStyle(f::ComposedFunction) =
    _combine_batched(BatchedMulStyle(f.outer), BatchedMulStyle(f.inner))
_combine_batched(::BatchedMul, ::BatchedMul) = BatchedMul()
_combine_batched(::BatchedMulStyle, ::BatchedMulStyle) = ColumnwiseMul()

# Generic batched application: native kernel or column-wise fallback,
# selected by BatchedMulStyle. Composite operators batch natively by
# composition, MulFuncOperator forwards the style of its ovp function.
explicit_mul_impl(op::MatrixShapedOperator, X::AbstractMatrix{<:Number}) =
    _explicit_mul_impl(BatchedMulStyle(op), op, X)

_explicit_mul_impl(::BatchedMul, op, X) = batched_mul_impl(op, X)
_explicit_mul_impl(::ColumnwiseMul, op, X) = _mapcols(op, X)

# Subtypes only have to implement vector application, so batched
# application of operators falls back to column-wise:
batched_mul_impl(op::MatrixShapedOperator, X::AbstractMatrix{<:Number}) = _mapcols(op, X)

# Column-wise fallback kernel; reduce(hcat, ...) is the AD-proven form,
# a stack over eachcol may replace it after AD/GPU verification:
_mapcols(op, X::AbstractMatrix) = reduce(hcat, [explicit_mul_impl(op, X[:, j]) for j in axes(X, 2)])


# is* on operators defaults to the structure traits; for real element
# types symmetric and hermitian coincide:
LinearAlgebra.issymmetric(op::MatrixShapedOperator{T}) where T =
    Symmetricity(op) isa IsSymmetric || NumberDomain(T) === Real && Hermitianity(op) isa IsHermitian
LinearAlgebra.ishermitian(op::MatrixShapedOperator{T}) where T =
    Hermitianity(op) isa IsHermitian || NumberDomain(T) === Real && Symmetricity(op) isa IsSymmetric
LinearAlgebra.isposdef(op::MatrixShapedOperator) = PosDefNess(op) isa IsPosDef


# Internal value-refined psd check, used by composite isposdef theorems
# (all-psd terms plus one posdef term make a posdef sum):
_ispossemidef(A) = PosDefNess(A) isa IsPosSemiDef || isposdef(A)
_ispossemidef(A::Diagonal) = all(x -> isreal(x) && real(x) >= 0, A.diag)
_ispossemidef(J::UniformScaling) = isreal(J.λ) && real(J.λ) >= 0


# Draft note: value-refined `isposdef` methods (non-zero diagonal of
# structurally triangular Gram factors, block-diagonal compositions) to
# be added at implementation time.
