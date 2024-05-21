include("test_util.jl")

@testset verbose=true "PointNeighbors.jl Tests" begin
    include("nhs_trivial.jl")
    include("nhs_grid.jl")
end
