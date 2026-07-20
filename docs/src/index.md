# MatrixShapedOperators.jl

Linear operators with matrix shape and semantics, accessed through
application and operator algebra instead of element access: operators
combine lazily with operators and eagerly with arrays. The design is
tracing- and GPU-friendly, built on whole-array operations.

## Concepts

Operators combine lazily with operators and eagerly with arrays:
`op1 * op2` and `op1 + op2` (as well as `-`, `/` and scalar scaling)
build lazy [`MatrixShapedProduct`](@ref)s and
[`MatrixShapedSum`](@ref)s, while `op * x` and `op * X` apply an
operator eagerly to a vector resp. to the columns of a matrix, and
`op \ x` solves against vectors or multiple right-hand sides.
Matrices enter the lazy algebra via [`asoperator`](@ref) and leave it
via [`asmatrix`](@ref) where a cheap matrix representation exists;
`Matrix(op)` and `AbstractMatrix(op)` materialize any operator
explicitly.

Structural properties are captured at construction time and carried in
the type domain as trait singletons: `Symmetricity(op)`,
`Hermitianity(op)`, `PosDefNess(op)`, `Triangularity(op)`,
`Unitarity(op)` and `RowRankNess(op)` answer `Is*()` or `Unknown*()`.
`Is*` answers hold for any element values, so they are safe to branch
on under program tracing. The `is*` functions (`issymmetric`,
`isposdef`, ...) default to the traits and add cheap value-based
refinements; `true` is definitive, `false` may just mean the property
is not derivable. Traits propagate through the operator algebra and
are extensible by third-party packages.

Operators are constructed from matrices via [`asoperator`](@ref) and
[`diagonal_operator`](@ref), from multiplication functions (e.g. from
autodiff) via [`mulfunc_operator`](@ref), and from other operators via
[`rowgram_operator`](@ref) and [`colgram_operator`](@ref) (`A * A'`
resp. `A' * A`, storing only `A`), [`woodbury_operator`](@ref)
(`A + B * D * B'`) and [`blockdiag_operator`](@ref). Adding the
sizeless identity [`𝟙`](@ref) or a `LinearAlgebra.I` (`λ * I` in
general) contributes an identity resp. uniform-scaling term, while
multiplying by `𝟙` and adding `zero(op)` are no-ops at the type level.

Structural operations are defined where the operator structure
supports them, without ever materializing or factorizing implicit
operators. [`lower_cholesky`](@ref) and [`upper_cholesky`](@ref) give
triangular Cholesky factors, [`rowgram_factor`](@ref) and
[`colgram_factor`](@ref) the loose `F * F'` resp. `G' * G` factors used
for sampling and whitening, `logabsdet` the log-determinant and
`op \ x` the solve. They compose through scalings, products,
block-diagonals and Woodbury operators, but not through sums;
`LinearAlgebra.factorize` caches a Woodbury factorization for repeated
solves and determinants.
[`MatrixShapedOperators.test_operator`](@ref) tests the operator
contract of custom operator implementations.

Package extensions provide conversion to
[LinearMaps](https://github.com/JuliaLinearAlgebra/LinearMaps.jl) and
[SciMLOperators](https://github.com/SciML/SciMLOperators.jl) operator
types, [Statistics](https://github.com/JuliaStats/Statistics.jl)
`mean` of operator collections,
[ChainRulesCore](https://github.com/JuliaDiff/ChainRulesCore.jl)
non-differentiability declarations for the trait machinery,
[Adapt](https://github.com/JuliaGPU/Adapt.jl)-based storage adaption,
[InverseFunctions](https://github.com/JuliaMath/InverseFunctions.jl)
declarations for the Gram constructor/factor pairs, and
[SparseArrays](https://github.com/JuliaSparse/SparseArrays.jl) support
(sparse block-diagonal collapse and explicit `sparse(op)`
materialization).
