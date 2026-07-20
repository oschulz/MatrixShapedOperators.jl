# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).


# Adding a sizeless UniformScaling adds a sized scaled-identity term,
# the checked `+` funnel rejects non-square operators:
Base.:(+)(a::MatrixShapedOperator, J::UniformScaling) = a + J.λ * IdentityOperator(size(a, 1))
Base.:(+)(J::UniformScaling, a::MatrixShapedOperator) = a + J


"""
    struct MatrixShapedSum{T<:Number,OPS<:Union{Tuple,AbstractVector}} <: MatrixShapedOperator{T}

Represents the sum `terms[1] + terms[2] + ...` of equal-size
matrix-shaped operators.

User code should not call `MatrixShapedSum` directly, but use `+` instead:

```julia
op_a + op_b + 2 * I isa MatrixShapedSum
```

# Extended help

Adding a `UniformScaling` adds a [`UniformScalingOperator`](@ref) term.
Terms may be a tuple (few, heterogeneous) or a vector (many, uniform),
[`MatrixShaped`](@ref) values are normalized via [`asoperator`](@ref).

The sum is symmetric/hermitian/positive semi-definite if all terms are,
and positive definite if additionally at least one term is.
"""
struct MatrixShapedSum{T<:Number,OPS<:Union{Tuple,AbstractVector}} <: MatrixShapedOperator{T}
    terms::OPS

    function MatrixShapedSum{T,OPS}(terms::OPS) where {T<:Number,OPS<:Union{Tuple,AbstractVector}}
        isempty(terms) && throw(ArgumentError("MatrixShapedSum requires at least one term"))
        sz = size(first(terms))
        all(t -> size(t) == sz, terms) || throw(DimensionMismatch(
            "MatrixShapedSum terms must all have equal size"
        ))
        return new{T,OPS}(terms)
    end
end
export MatrixShapedSum

function MatrixShapedSum(terms::Union{Tuple,AbstractVector})
    isempty(terms) && throw(ArgumentError("MatrixShapedSum requires at least one term"))
    ops = map(asoperator, terms)
    T = mapreduce(eltype, promote_type, ops)
    return MatrixShapedSum{T,typeof(ops)}(ops)
end

Base.size(op::MatrixShapedSum) = size(first(op.terms))

# Structure traits via the trait algebra:
traitsof(op::MatrixShapedSum) = mapreduce(traitsof, traitset_add, op.terms)
for C in (:Symmetricity, :Hermitianity, :PosDefNess, :Triangularity, :Unitarity, :RowRankNess)
    @eval $C(op::MatrixShapedSum) = gettrait(traitsof(op), $C)
end

LinearAlgebra.issymmetric(op::MatrixShapedSum) = all(issymmetric, op.terms)
LinearAlgebra.ishermitian(op::MatrixShapedSum) = all(ishermitian, op.terms)
LinearAlgebra.isposdef(op::MatrixShapedSum) = all(_ispossemidef, op.terms) && any(isposdef, op.terms)

Base.adjoint(op::MatrixShapedSum) = MatrixShapedSum(map(adjoint, op.terms))

explicit_mul_impl(op::MatrixShapedSum, v::AbstractVector{<:Number}) =
    mapreduce(t -> explicit_mul_impl(t, v), +, op.terms)
batched_mul_impl(op::MatrixShapedSum, X::AbstractMatrix{<:Number}) =
    mapreduce(t -> explicit_mul_impl(t, X), +, op.terms)

Base.:(==)(a::MatrixShapedSum, b::MatrixShapedSum) = _parts_equal(a.terms, b.terms)

Base.hash(op::MatrixShapedSum, h::UInt) = _parts_hash(op.terms, hash(:MatrixShapedSum, h))

Base.isapprox(a::MatrixShapedSum, b::MatrixShapedSum; kwargs...) =
    length(a.terms) == length(b.terms) &&
    all(isapprox(ta, tb; kwargs...) for (ta, tb) in zip(a.terms, b.terms))

function Base.show(io::IO, op::MatrixShapedSum)
    print(io, "MatrixShapedSum(")
    show(io, op.terms)
    print(io, ")")
end

_vec_terms(terms::Tuple) = [terms...]
_vec_terms(terms::AbstractVector) = terms

# Equality and hashing of term/factor collections must not depend on
# tuple vs vector storage:
_parts_equal(a, b) = length(a) == length(b) && all(x == y for (x, y) in zip(a, b))
_parts_hash(parts, h::UInt) = foldl((h, p) -> hash(p, h), parts; init = h)

_cat_terms(a::Tuple, b::Tuple) = (a..., b...)
_cat_terms(a, b) = vcat(_vec_terms(a), _vec_terms(b))

lazy_add_impl(a::MatrixShapedOperator, b::MatrixShapedOperator) = MatrixShapedSum((a, b))
lazy_add_impl(a::MatrixShapedSum, b::MatrixShapedOperator) = MatrixShapedSum(_cat_terms(a.terms, (b,)))
lazy_add_impl(a::MatrixShapedOperator, b::MatrixShapedSum) = MatrixShapedSum(_cat_terms((a,), b.terms))
lazy_add_impl(a::MatrixShapedSum, b::MatrixShapedSum) = MatrixShapedSum(_cat_terms(a.terms, b.terms))


"""
    struct MatrixShapedProduct{T<:Number,OPS<:Union{Tuple,AbstractVector}} <: MatrixShapedOperator{T}

Represents the product `factors[1] * factors[2] * ...` of matrix-shaped
operators with matching adjacent sizes, applied right-to-left.

User code should not call `MatrixShapedProduct` directly, but use `*` instead:

```julia
op_a * op_b isa MatrixShapedProduct
```

# Extended help

Factors may be a tuple (few, heterogeneous) or a vector (many, uniform),
[`MatrixShaped`](@ref) values are normalized via [`asoperator`](@ref).

Symmetry/definiteness traits are not preserved by products; use
[`RowGramOperator`](@ref) for `A * A'`. Triangularity, unitarity and
full row rank compose (see the trait algebra).
"""
struct MatrixShapedProduct{T<:Number,OPS<:Union{Tuple,AbstractVector}} <: MatrixShapedOperator{T}
    factors::OPS

    function MatrixShapedProduct{T,OPS}(factors::OPS) where {T<:Number,OPS<:Union{Tuple,AbstractVector}}
        isempty(factors) && throw(ArgumentError("MatrixShapedProduct requires at least one factor"))
        for i in firstindex(factors):(lastindex(factors) - 1)
            size(factors[i], 2) == size(factors[i + 1], 1) || throw(DimensionMismatch(
                "operator of size $(size(factors[i])) can't be composed with operator of size $(size(factors[i + 1]))"
            ))
        end
        return new{T,OPS}(factors)
    end
end
export MatrixShapedProduct

function MatrixShapedProduct(factors::Union{Tuple,AbstractVector})
    isempty(factors) && throw(ArgumentError("MatrixShapedProduct requires at least one factor"))
    ops = map(asoperator, factors)
    T = mapreduce(eltype, promote_type, ops)
    return MatrixShapedProduct{T,typeof(ops)}(ops)
end

Base.size(op::MatrixShapedProduct) = (size(first(op.factors), 1), size(last(op.factors), 2))

# Structure traits via the trait algebra (triangularity, unitarity and
# full row rank compose under products, symmetry/definiteness do not):
traitsof(op::MatrixShapedProduct) = mapreduce(traitsof, traitset_mul, op.factors)
for C in (:Symmetricity, :Hermitianity, :PosDefNess, :Triangularity, :Unitarity, :RowRankNess)
    @eval $C(op::MatrixShapedProduct) = gettrait(traitsof(op), $C)
end

Base.adjoint(op::MatrixShapedProduct) = MatrixShapedProduct(reverse(map(adjoint, op.factors)))

explicit_mul_impl(op::MatrixShapedProduct, v::AbstractVector{<:Number}) =
    foldr(explicit_mul_impl, op.factors; init = v)
batched_mul_impl(op::MatrixShapedProduct, X::AbstractMatrix{<:Number}) =
    foldr(explicit_mul_impl, op.factors; init = X)

Base.:(==)(a::MatrixShapedProduct, b::MatrixShapedProduct) = _parts_equal(a.factors, b.factors)

Base.hash(op::MatrixShapedProduct, h::UInt) = _parts_hash(op.factors, hash(:MatrixShapedProduct, h))

Base.isapprox(a::MatrixShapedProduct, b::MatrixShapedProduct; kwargs...) =
    length(a.factors) == length(b.factors) &&
    all(isapprox(fa, fb; kwargs...) for (fa, fb) in zip(a.factors, b.factors))

function Base.show(io::IO, op::MatrixShapedProduct)
    print(io, "MatrixShapedProduct(")
    show(io, op.factors)
    print(io, ")")
end

# The determinant of a product of square factors is the product of the
# factor determinants:
function LinearAlgebra.logabsdet(op::MatrixShapedProduct)
    all(f -> size(f, 1) == size(f, 2), op.factors) || throw(ArgumentError(
        "no structural logabsdet available for an operator product with non-square factors"
    ))
    parts = map(logabsdet, op.factors)
    return sum(first, parts), prod(last, parts)
end

# Inverting a product of square factors solves factor by factor, left
# to right:
function ldiv_impl(op::MatrixShapedProduct, x::AbstractVecOrMat{<:Number})
    all(f -> size(f, 1) == size(f, 2), op.factors) || throw(ArgumentError(
        "no structural inverse application available for an operator product with non-square factors"
    ))
    return foldl((v, f) -> ldiv_impl(f, v), op.factors; init = x)
end

lazy_mul_impl(a::MatrixShapedOperator, b::MatrixShapedOperator) = MatrixShapedProduct((a, b))
lazy_mul_impl(a::MatrixShapedProduct, b::MatrixShapedOperator) = MatrixShapedProduct(_cat_terms(a.factors, (b,)))
lazy_mul_impl(a::MatrixShapedOperator, b::MatrixShapedProduct) = MatrixShapedProduct(_cat_terms((a,), b.factors))
lazy_mul_impl(a::MatrixShapedProduct, b::MatrixShapedProduct) = MatrixShapedProduct(_cat_terms(a.factors, b.factors))
