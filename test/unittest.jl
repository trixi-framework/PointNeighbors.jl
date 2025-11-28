# Separate file that can be executed to only run unit tests.
# Include `test_util.jl` first.
@testset verbose=true "Unit Tests" begin
    include("vector_of_vectors.jl")
    include("nhs_trivial.jl")
    include("nhs_grid.jl")
    include("nhs_precomputed.jl")
    include("neighborhood_search.jl")
    include("cell_lists/full_grid.jl")
end;
