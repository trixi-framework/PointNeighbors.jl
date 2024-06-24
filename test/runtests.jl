include("test_util.jl")

@testset verbose=true "PointNeighbors.jl Tests" begin
    include("vector_of_vectors.jl")
    include("nhs_trivial.jl")
    include("nhs_grid.jl")
    include("neighborhood_search.jl")
end;
