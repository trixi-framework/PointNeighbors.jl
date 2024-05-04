include("test_util.jl")

@testset verbose=true "TrixiNeighborhoodSearch.jl Tests" begin
    include("trivial_nhs.jl")
    include("grid_nhs.jl")
end
