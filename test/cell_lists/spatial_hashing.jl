@testset verbose=true "SpatialHashingCellList" begin
    # General test for 2D and 3D
    @testset verbose=true "Compare Against `TrivialNeighborhoodSearch`" begin
        cloud_sizes = [
            (10, 11),
            (100, 90),
            (8, 10, 6),
            (39, 40, 41)
        ]
        name(size) = "$(length(size))D with $(prod(size)) Particles"
        @testset verbose=true "$(name(cloud_size))" for cloud_size in cloud_sizes
            coords = point_cloud(cloud_size, seed = 1)
            NDIMS = length(cloud_size)
            n_points = size(coords, 2)
            search_radius = 2.5

            # Compute expected neighbor lists by brute-force looping over all points
            # as potential neighbors (`TrivialNeighborhoodSearch`).
            trivial_nhs = TrivialNeighborhoodSearch{NDIMS}(; search_radius,
                                                           eachpoint = axes(coords, 2))

            neighbors_expected = [Int[] for _ in axes(coords, 2)]

            foreach_point_neighbor(coords, coords, trivial_nhs,
                                   parallel = false) do point, neighbor,
                                                        pos_diff, distance
                append!(neighbors_expected[point], neighbor)
            end

            nhs = GridNeighborhoodSearch{NDIMS}(; search_radius, n_points,
                                                cell_list = SpatialHashingCellList{NDIMS}(2 *
                                                                                      n_points))

            initialize!(nhs, coords, coords)

            # Test if there are any collisions.
            @testset verbose=true "Collision Detection" begin
                @test any(nhs.cell_list.collisions)
            end

            neighbors = [Int[] for _ in axes(coords, 2)]
            foreach_point_neighbor(coords, coords, nhs,
                                   parallel = false) do point, neighbor,
                                                        pos_diff, distance
                push!(neighbors[point], neighbor)
            end

            @test sort.(neighbors) == neighbors_expected
        end
    end

    # Test list behavior with empty cells
    @testset "Collision Handling with empty cells" begin
        # The point is in cell (-1, 0) which has a hash collision with cell (-2, -1)
        point = [-0.05, 0.05]
        coordinates = hcat(point)
        NDIMS = size(coordinates, 1)
        n_points = size(coordinates, 2)
        search_radius = 0.1 + 10 * eps()
        point_index = 1

        nhs = GridNeighborhoodSearch{2}(; search_radius, n_points,
                                            cell_list = SpatialHashingCellList{NDIMS}(n_points))
        initialize_grid!(nhs, coordinates)

        found_neighbors = Int[]
        foreach_neighbor(coordinates, coordinates, nhs,
                            point_index) do point, neighbor, pos_diff, distance
            push!(found_neighbors, neighbor)
        end

        correct_neighbors = [1]
        @test correct_neighbors == found_neighbors
    end
end
