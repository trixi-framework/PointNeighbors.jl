@testset verbose=true "TrivialNeighborhoodSearch" begin
    # Setup with 5 points
    nhs = TrivialNeighborhoodSearch{2}(search_radius = 1.0, eachpoint = Base.OneTo(5))

    # Get each neighbor for arbitrary coordinates
    neighbors = collect(PointNeighbors.eachneighbor([1.0, 2.0], nhs))

    #### Verification
    @test neighbors == [1, 2, 3, 4, 5]

    @testset "`eachindex_y` Overrides `eachpoint`" begin
        # 10 neighbor points, of which only the points 3 to 7 are considered
        y = rand(2, 10)
        nhs = TrivialNeighborhoodSearch{2}(search_radius = 1.0)

        initialize!(nhs, y, y, eachindex_y = 3:7)
        @test collect(PointNeighbors.eachneighbor([1.0, 2.0], nhs)) == 3:7

        update!(nhs, y, y, eachindex_y = 2:4)
        @test collect(PointNeighbors.eachneighbor([1.0, 2.0], nhs)) == 2:4

        # Without `eachindex_y`, all points in `y` are considered
        update!(nhs, y, y)
        @test collect(PointNeighbors.eachneighbor([1.0, 2.0], nhs)) == 1:10
    end
end
