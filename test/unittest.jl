# Separate file that can be executed to only run unit tests.
# Include `test_util.jl` first.
@testset verbose=true "Unit Tests" begin
    include("nhs_trivial.jl")
    include("nhs_grid.jl")
    include("neighborhood_search.jl")
end;
