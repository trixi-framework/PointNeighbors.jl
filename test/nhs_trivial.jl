@testset verbose=true "TrivialNeighborhoodSearch" begin
    # Setup with 5 points
    nhs = TrivialNeighborhoodSearch{2}(search_radius = 1.0, eachpoint = Base.OneTo(5))

    # Get each neighbor for arbitrary coordinates
    neighbors = collect(PointNeighbors.eachneighbor([1.0, 2.0], nhs))

    #### Verification
    @test neighbors == [1, 2, 3, 4, 5]
end
