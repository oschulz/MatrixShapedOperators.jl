# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).

using Test
using TimerOutputs: @timed_testset
using MatrixShapedOperators
import Documenter

Documenter.DocMeta.setdocmeta!(
    MatrixShapedOperators,
    :DocTestSetup,
    :(using MatrixShapedOperators);
    recursive = true
)
Documenter.doctest(MatrixShapedOperators)
