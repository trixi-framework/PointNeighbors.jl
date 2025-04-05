@testset verbose=true "SpatialHashNeighborhoodSearch" begin
    @testset "Rectangular Point Cloud 2D" begin
        #### Setup
        # Rectangle of equidistantly spaced points
        # from (x, y) = (-0.25, -0.25) to (x, y) = (0.35, 0.35).
        range = -0.25:0.1:0.35
        coordinates1 = hcat(collect.(Iterators.product(range, range))...)
        n_points = size(coordinates1, 2)

        point_position1 = [0.05, 0.05]
        search_radius = 0.1

        # Create neighborhood search
        nhs1 = GridNeighborhoodSearch{2}(; search_radius, n_points,
                                         cell_list = SpatialHashingCellList{2}(2 * n_points))
        initialize_grid!(nhs1, coordinates1)

        # Get each neighbor for `point_position1`
        neighbors1 = sort(collect(PointNeighbors.eachneighbor(point_position1, nhs1)))

        # Move points and update NHS
        coordinates2 = coordinates1 .+ [1.4, -3.5]
        empty!(nhs1.cell_list)
        initialize_grid!(nhs1, coordinates2)

        # Get each neighbor for updated NHS
        neighbors2 = sort(collect(PointNeighbors.eachneighbor(point_position1, nhs1)))

        # Change position
        point_position2 = point_position1 .+ [1.4, -3.5]

        # Get each neighbor for `point_position2`
        neighbors3 = sort(collect(PointNeighbors.eachneighbor(point_position2, nhs1)))

        # Double search radius
        nhs2 = GridNeighborhoodSearch{2}(search_radius = 2 * search_radius,
                                         n_points = size(coordinates1, 2),
                                         cell_list = SpatialHashingCellList{2}(2 * n_points))
        initialize!(nhs2, coordinates1, coordinates1)

        # Get each neighbor in double search radius
        neighbors4 = sort(collect(PointNeighbors.eachneighbor(point_position1, nhs2)))

        # Move points
        coordinates2 = coordinates1 .+ [0.4, -0.4]
        empty!(nhs2.cell_list)
        initialize_grid!(nhs2, coordinates2)

        # Get each neighbor in double search radius
        neighbors5 = sort(collect(PointNeighbors.eachneighbor(point_position1, nhs2)))

        #### Verification against lists of potential neighbors built by hand
        correct_neighbors1 = [17, 18, 19, 24, 25, 26, 31, 32, 33]
        @test all(x -> x in neighbors1, correct_neighbors1)

        correct_neighbors2 = Int[]
        @test all(x -> x in neighbors2, correct_neighbors2)

        correct_neighbors3 = [17, 18, 19, 24, 25, 26, 31, 32, 33]
        @test all(x -> x in neighbors3, correct_neighbors3)

        correct_neighbors4 = [
            9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 30, 31,
            32, 33, 34, 35, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49]
        @test all(x -> x in neighbors4, correct_neighbors4)

        correct_neighbors5 = [36, 37, 38, 43, 44, 45]
        @test all(x -> x in neighbors5, correct_neighbors5)
    end
end;
