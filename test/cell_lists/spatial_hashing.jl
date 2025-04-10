@testset verbose=true "SpatialHashingCellList" begin
    # General test for 2D and 3D
    @testset verbose=true "Compare Against `TrivialNeighborhoodSearch`" begin
        cloud_sizes = [
            (10, 11),
            # (100, 90),
            # (3, 3, 3),
            # (39, 40, 41)
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

            @testset verbose=true "Empty Cell and Collision Detection" begin
                # 1. Find empty lists
                # 2. Check for collisions with non-empty lists
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

    # # Explicit test, how the data structure handels collisions (very simple)
    # @testset "Collisions with foreach_neighbor" begin
    #     range = -0.35:0.1:0.25
    #     coordinates1 = hcat(collect.(Iterators.product(range, range))...)
    #     n_points = size(coordinates1, 2)
    #     search_radius = 0.1 + 10 * eps()
    #     point_index1 = 25
    #     point_position1 = coordinates1[point_index1]

    #     @testset verbose=true "List Size $(list_size)" for list_size in [
    #         2 * n_points,
    #         1,
    #         n_points,
    #         4 * n_points
    #     ]
    #         nhs1 = GridNeighborhoodSearch{2}(; search_radius, n_points,
    #                                          cell_list = SpatialHashingCellList{2}(list_size))
    #         initialize_grid!(nhs1, coordinates1)

    #         @testset verbose=true "Collision at test point" begin
    #             cell = PointNeighbors.cell_coords(point_position1, nhs1)
    #             index = PointNeighbors.spatial_hash(cell, nhs1.cell_list.list_size)
    #             @test nhs1.cell_list.collisions[index] == true
    #         end

    #         found_neighbors1 = Int[]
    #         foreach_neighbor(coordinates1, coordinates1, nhs1,
    #                          point_index1) do point, neighbor, pos_diff, distance
    #             push!(found_neighbors1, neighbor)
    #         end

    #         correct_neighbors1 = [18, 24, 25, 26, 32]
    #         @test correct_neighbors1 == found_neighbors1
    #     end
    # end
    @testset "Collision Handling with empty cells" begin
        #TODO: Add explicit test for list behavior with empty cells
    end
end
