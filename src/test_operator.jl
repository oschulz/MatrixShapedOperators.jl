# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).


"""
    MatrixShapedOperators.test_operator(
        op::MatrixShapedOperator;
        directions = :both, x, y, u, compare = isapprox, materialize = true, kwargs...
    )

Tests the operator contract of `op` via `Test.@test`: linearity,
adjoint consistency, batched against column-wise application and, with
`materialize = true` (default), agreement of `Matrix(op)` with
application and with the declared structure traits.

Meant for the test suites of operator implementations, in particular
of [`mulfunc_operator`](@ref) seams. Probe arrays default to
deterministic values and can be given via the keyword arguments `x`,
`y` (operator domain) and `u` (operator codomain); `compare` (default
`isapprox`) and further keyword arguments are forwarded to the
comparison.

For one-directional operators `directions = :forward` skips the
adjoint-consistency test and `directions = :adjoint` tests
`adjoint(op)` that way instead; probe arrays refer to the tested
direction.

!!! Note
    Requires the `Test` standard library to be loaded.
"""
function test_operator end
@compat public test_operator
