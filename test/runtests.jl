# This file is a part of MatrixShapedOperators.jl, licensed under the MIT License (MIT).

import Test

Test.@testset "Package MatrixShapedOperators" begin
    include("test_aqua.jl")
    include("test_matrix_shaped.jl")
    include("test_docs.jl")
end # testset
