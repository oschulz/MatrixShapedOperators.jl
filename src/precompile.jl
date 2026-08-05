# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).

import PrecompileTools

# Woodbury operators are deliberately left out, precompilation increases
# package load time a lot.

PrecompileTools.@compile_workload begin
    for T in (Float64, Float32)
        A = T[1 2; 3 4]
        S = T[5 1; 1 4]
        d = T[1, 2]
        x = T[1, 2]
        X = T[1 2; 3 4]

        op = asoperator(A)
        dop = diagonal_operator(d)
        pop = asoperator(Symmetric(S), IsPosDef())
        ltop = asoperator(LowerTriangular(S))

        # application: vector, batched, empty batch, adjoint, mul!:
        op * x
        op * X
        op * similar(X, T, 2, 0)
        adjoint(op) * x
        mul!(similar(x), op, x)
        mul!(similar(x), op, x, T(2), T(0))

        # lazy operator algebra:
        (T(2) * op) * x
        (-op) * x
        (op + dop + I) * x
        (op * adjoint(dop)) * x
        (op + 𝟙) * x
        zero(op) * x
        (T(2) * 𝟙(2)) * x
        𝟙(2) * x

        # structured and composite operators:
        blockdiag_operator(op, dop) * vcat(x, x)
        blockdiag_operator(dop, T(2) * 𝟙(2)) * vcat(x, x)
        g = rowgram_operator(A)
        g * x
        colgram_operator(A) * x

        # structural solves, determinants and factors:
        pop \ x
        ltop \ x
        logabsdet(pop)
        lower_cholesky(pop)
        lower_cholesky(rowgram_operator(LowerTriangular(S)))
        isposdef(g)

        # materialization:
        Matrix(op)
        AbstractMatrix(g)
    end
end
