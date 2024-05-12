include("test_util.jl")

@testset verbose=true "PointNeighbors.jl Tests" begin
    include("trivial_nhs.jl")
    include("grid_nhs.jl")
    include("neighborhood_search.jl")
end;
