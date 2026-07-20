# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).


"""
    mulfunc_operator(::Type{T}, sz::Dims{2}, ovp, vop, decls...)
    mulfunc_operator(::Type{OP}, ::Type{T}, sz::Dims{2}, ovp, vop, decls...)

Returns a matrix-shaped operator with element type `T` and size `sz`,
defined by a multiplication function `ovp` and an adjoint
multiplication function `vop`. Either may be `nothing`, the missing
direction then fails on application; materialization (`Matrix` target)
and conversion to foreign operator types require `ovp`.

Typically returns a [`MulFuncOperator`](@ref):

```julia
op = mulfunc_operator(T, sz, ovp, vop)
eltype(op) == T
size(op) == sz
op * x_r == ovp(x_r)
op' * x_l == vop(x_l)
```

# Extended help

Structural properties, which the multiplication functions can't be
inspected for, are declared via trait values `decls...` (see
[`MatrixShapedOperators.StructureTrait`](@ref)), e.g.
`mulfunc_operator(T, sz, ovp, vop, IsHermitian())`; non-square
operators can only carry rank declarations. Declarations are trusted,
not verified - [`MatrixShapedOperators.test_operator`](@ref) checks
them numerically.

The form with a leading operator type `OP` builds that operator type
instead, including foreign types like `LinearMaps.LinearMap` (package
extensions) and `Matrix`, which materializes. It lets packages that
generate multiplication functions, e.g. autodiff packages, target a
requested operator type without depending on its package.
"""
function mulfunc_operator end
export mulfunc_operator

function mulfunc_operator(::Type{T}, sz::Dims{2}, ovp, vop, decls...) where {T<:Number}
    return MulFuncOperator{T}(ovp, vop, sz, decls...)
end


"""
    struct MulFuncOperator{T<:Number,TRS<:Tuple,F,G} <: MatrixShapedOperator{T}

Linear operator defined by a multiplication function `ovp` and an
adjoint multiplication function `vop`.

User code should not call `MulFuncOperator` directly, but use
[`mulfunc_operator`](@ref) instead.

# Extended help

The structure traits declared at construction are stored as a
[`traitset`](@ref) in the type parameter `TRS`.

`op * X` hands the whole matrix to `ovp` if
[`BatchedMulStyle`](@ref)`(ovp)` is [`BatchedMul`](@ref)`()`, and calls
`ovp` on each column separately otherwise (the default); likewise for
`op' * X` and `vop`. Scalar scaling drops the definiteness declaration.
Compatible with program tracing if `ovp` and `vop` are.
"""
struct MulFuncOperator{T<:Number,TRS<:Tuple,F,G} <: MatrixShapedOperator{T}
    ovp::F
    vop::G
    sz::Dims{2}

    function MulFuncOperator{T,TRS,F,G}(ovp, vop, sz::Dims{2}) where {T<:Number,TRS<:Tuple,F,G}
        all(>=(0), sz) || throw(ArgumentError(
            "MulFuncOperator size must be non-negative, got $sz"
        ))
        ts = _traits_instance(TRS)
        all(t -> t isa RowRankNess, ts) || sz[1] == sz[2] || throw(ArgumentError(
            "MulFuncOperator of size $sz can't carry square-only structure declarations"
        ))
        ovp === nothing && vop === nothing && throw(ArgumentError(
            "MulFuncOperator requires at least one multiplication function"
        ))
        return new{T,TRS,F,G}(ovp, vop, sz)
    end
end
export MulFuncOperator

function MulFuncOperator{T}(ovp::F, vop::G, sz::Dims{2}, decls...) where {T<:Number,F,G}
    ts = traitset(decls...)
    return MulFuncOperator{T,typeof(ts),F,G}(ovp, vop, sz)
end

function mulfunc_operator(::Type{OP}, ::Type{T}, sz::Dims{2}, ovp, vop, decls...) where {OP<:MulFuncOperator,T<:Number}
    return MulFuncOperator{T}(ovp, vop, sz, decls...)
end

function mulfunc_operator(::Type{Matrix}, ::Type{T}, sz::Dims{2}, ovp, vop, decls...) where {T<:Number}
    return convert(Matrix{T}, Matrix(MulFuncOperator{T}(ovp, vop, sz, decls...)))
end

Base.size(op::MulFuncOperator) = op.sz

traitsof(::MulFuncOperator{T,TRS}) where {T,TRS} = _traits_instance(TRS)

for C in (:Symmetricity, :Hermitianity, :PosDefNess, :Triangularity, :Unitarity, :RowRankNess)
    @eval $C(::MulFuncOperator{T,TRS}) where {T,TRS} = gettrait(TRS, $C)
end

BatchedMulStyle(op::MulFuncOperator) = BatchedMulStyle(op.ovp)

function Base.adjoint(op::MulFuncOperator{T,TRS,F,G}) where {T,TRS,F,G}
    ts = traitset_adjoint(_traits_instance(TRS))
    return MulFuncOperator{T,typeof(ts),G,F}(op.vop, op.ovp, reverse(op.sz))
end

explicit_mul_impl(op::MulFuncOperator, v::AbstractVector{<:Number}) = op.ovp(v)
batched_mul_impl(op::MulFuncOperator, X::AbstractMatrix{<:Number}) = op.ovp(X)

# A missing multiplication function fails on application, so operators
# without one direction construct and adjoint freely:
explicit_mul_impl(::MulFuncOperator{<:Number,<:Tuple,Nothing}, ::AbstractVector{<:Number}) =
    throw(ArgumentError("no multiplication function available for this direction of the operator"))
batched_mul_impl(::MulFuncOperator{<:Number,<:Tuple,Nothing}, ::AbstractMatrix{<:Number}) =
    throw(ArgumentError("no multiplication function available for this direction of the operator"))

# Trait declarations are type-level metadata and do not participate in
# equality, approximate equality or hashing:

Base.:(==)(a::MulFuncOperator, b::MulFuncOperator) =
    eltype(a) == eltype(b) && a.ovp == b.ovp && a.vop == b.vop && a.sz == b.sz

Base.hash(op::MulFuncOperator{T}, h::UInt) where T =
    hash(op.sz, hash(op.vop, hash(op.ovp, hash(T, hash(:MulFuncOperator, h)))))

Base.isapprox(a::MulFuncOperator, b::MulFuncOperator; kwargs...) = a == b

function Base.show(io::IO, op::MulFuncOperator{T}) where T
    print(io, "mulfunc_operator(", T, ", ")
    show(io, op.sz)
    print(io, ", ")
    show(io, op.ovp)
    print(io, ", ")
    show(io, op.vop)
    for t in traitsof(op)
        print(io, ", ")
        show(io, t)
    end
    print(io, ")")
end
