# MatrixShapedOperators.jl

A Julia package for linear operators with matrix shape and semantics.

[![Documentation for stable version](https://img.shields.io/badge/docs-stable-blue.svg)](https://oschulz.github.io/MatrixShapedOperators.jl/stable)
[![Documentation for development version](https://img.shields.io/badge/docs-dev-blue.svg)](https://oschulz.github.io/MatrixShapedOperators.jl/dev)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)
[![Build Status](https://github.com/oschulz/MatrixShapedOperators.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/oschulz/MatrixShapedOperators.jl/actions/workflows/CI.yml)
[![Codecov](https://codecov.io/gh/oschulz/MatrixShapedOperators.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/oschulz/MatrixShapedOperators.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

## Documentation

* [Documentation for stable version](https://oschulz.github.io/MatrixShapedOperators.jl/stable)
* [Documentation for development version](https://oschulz.github.io/MatrixShapedOperators.jl/dev)

## Related packages

[LinearMaps.jl](https://github.com/JuliaLinearAlgebra/LinearMaps.jl),
[SciMLOperators.jl](https://github.com/SciML/SciMLOperators.jl) and
[LinearOperators.jl](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl)
also represent linear operators beyond explicit matrices.
MatrixShapedOperators.jl differs in that its operators are immutable
and applied out-of-place on whole arrays, carry their structural
properties (hermitianity, definiteness, triangularity, rank) in the
type domain, and provide structural operations on that basis: Cholesky
and Gram factors, determinants and solves.
