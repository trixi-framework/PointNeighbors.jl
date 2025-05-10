# All `using` calls are in this file, so that one can run any test file
# after running only this file.
using Test: @test, @testset, @test_throws
using TrixiTest: @trixi_test_nowarn
using PointNeighbors

"""
    @trixi_testset "name of the testset" #= code to test #=

Similar to `@testset`, but wraps the code inside a temporary module to avoid
namespace pollution.
"""
macro trixi_testset(name, expr)
    @assert name isa String

    mod = gensym()

    # TODO: `@eval` is evil
    quote
        @eval module $mod
        using Test
        using PointNeighbors

        # We also include this file again to provide the definition of
        # the other testing macros. This allows to use `@trixi_testset`
        # in a nested fashion and also call `@test_nowarn_mod` from
        # there.
        include(@__FILE__)

        @testset verbose=true $name $expr
        end

        nothing
    end
end

include("point_cloud.jl")
