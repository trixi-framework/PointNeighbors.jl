@testset verbose=true "SpatialHashingCellList" begin
    @testset "Collision Handling With Empty Cells" begin
        # The point is in cell (-1, 0) which has a hash collision with cell (-2, -1)
        coordinates = [-0.05; 0.05;;]
        NDIMS = size(coordinates, 1)
        n_points = size(coordinates, 2)
        search_radius = 0.1 + 10 * eps()
        point_index = 1

        nhs = GridNeighborhoodSearch{2}(; search_radius, n_points,
                                        cell_list = SpatialHashingCellList{NDIMS}(n_points))
        initialize_grid!(nhs, coordinates)

        @testset "Test For Collision" begin
            cell1 = (-1, 0)
            cell2 = (-2, -1)
            cell1_hash = PointNeighbors.spatial_hash(cell1, n_points)
            cell2_hash = PointNeighbors.spatial_hash(cell2, n_points)
            points1 = nhs.cell_list[cell1]
            points2 = nhs.cell_list[cell2]

            @test points1 == points2 == [1]
            @test cell1_hash == cell2_hash
        end

        neighbors = Int[]
        foreach_neighbor(coordinates, coordinates, nhs,
                         point_index) do point, neighbor, pos_diff, distance
            push!(neighbors, neighbor)
        end

        @test neighbors == [1]
    end

    @testset "Collision Handling With Non-Empty Cells" begin
        # Cell (-1, 0) with point 1 has a hash collision with cell (-2, -1) with point 2
        coordinates = [[-0.05 -0.15]; [0.05 -0.05]]
        NDIMS = size(coordinates, 1)
        n_points = size(coordinates, 2)
        search_radius = 0.1 + 10 * eps()
        point_index = 1

        nhs = GridNeighborhoodSearch{2}(; search_radius, n_points,
                                        cell_list = SpatialHashingCellList{NDIMS}(n_points))
        initialize_grid!(nhs, coordinates)

        @testset "Test For Collision" begin
            cell1 = (-1, 0)
            cell2 = (-2, -1)
            cell1_hash = PointNeighbors.spatial_hash(cell1, n_points)
            cell2_hash = PointNeighbors.spatial_hash(cell2, n_points)
            points1 = nhs.cell_list[cell1]
            points2 = nhs.cell_list[cell2]
            
            @test points1 == points2 == [1, 2]
            @test cell1_hash == cell2_hash
        end

        neighbors = [Int[] for _ in axes(coordinates, 2)]
        foreach_point_neighbor(coordinates, coordinates, nhs,
                               points = axes(coordinates, 2)) do point, neighbor, pos_diff,
                                                                 distance
            push!(neighbors[point], neighbor)
        end

        @test neighbors[1] == [1]
        @test neighbors[2] == [2]
    end
end
