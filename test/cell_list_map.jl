# Tests for `CellListMapNeighborhoodSearch`, defined in the package extension
# `PointNeighborsCellListMapExt`. `CellListMap` is loaded in `test_util.jl`.
@testset verbose=true "`CellListMapNeighborhoodSearch`" begin
    @testset "Compare Against `TrivialNeighborhoodSearch`" begin
        cloud_sizes = [
            (10, 11),
            (9, 10, 7)
        ]

        name(cloud_size,
             points_equal_neighbors) = "$(length(cloud_size))D with $(prod(cloud_size)) " *
                                       "Particles, `points_equal_neighbors = " *
                                       "$points_equal_neighbors`"

        @testset verbose=true "$(name(cloud_size, points_equal_neighbors))" for cloud_size in cloud_sizes,
                                                                                 points_equal_neighbors in (true,
                                                                                                           false)
            search_radius = 2.5
            coords = point_cloud(cloud_size, search_radius, seed = 1)
            NDIMS = length(cloud_size)
            n_points = size(coords, 2)

            # Different coordinates for `initialize!`, then `update!` with the actual
            # coordinates, to make sure `update!` (not just `initialize!`) works correctly.
            coords_initialize = point_cloud(cloud_size, search_radius, seed = 2)

            trivial_nhs = TrivialNeighborhoodSearch{NDIMS}(; search_radius,
                                                           eachpoint = axes(coords, 2))

            neighbors_expected = [Int[] for _ in axes(coords, 2)]
            foreach_point_neighbor(coords, coords, trivial_nhs,
                                   parallelization_backend = SerialBackend()) do point,
                                                                                 neighbor,
                                                                                 pos_diff,
                                                                                 distance
                push!(neighbors_expected[point], neighbor)
            end
            sort!.(neighbors_expected)

            nhs = CellListMapNeighborhoodSearch(NDIMS; search_radius, points_equal_neighbors)

            initialize!(nhs, coords_initialize, coords_initialize)
            update!(nhs, coords, coords)

            # `CellListMapNeighborhoodSearch` parallelizes by CellListMap.jl's own cell-pair
            # batches, not by splitting the `points` loop like every other NHS (see the
            # "Parallelization is not point-partitioned" warning on its docstring). So the
            # per-point `push!`/`@test` pattern below (unsynchronized, and safe with every
            # other NHS) is only valid with `SerialBackend()`; a given point index can be
            # visited from more than one thread with any other backend. We only check exact
            # per-point neighbor lists (and cross-check `pos_diff`/`distance`) serially, and
            # separately check just the (thread-safe, atomic) total pair count under
            # `PolyesterBackend()` to still exercise CellListMap.jl's parallel traversal.
            neighbors = [Int[] for _ in axes(coords, 2)]
            foreach_point_neighbor(coords, coords, nhs,
                                   parallelization_backend = SerialBackend()) do point, neighbor,
                                                                                  pos_diff,
                                                                                  distance
                push!(neighbors[point], neighbor)

                # Cross-check the returned `pos_diff` and `distance` against the
                # trivial definition.
                @test pos_diff ≈ coords[:, point] - coords[:, neighbor]
                @test distance ≈ sqrt(sum(abs2, pos_diff))
            end
            @test sort.(neighbors) == neighbors_expected

            n_pairs_expected = sum(length, neighbors_expected)
            n_pairs = Threads.Atomic{Int}(0)
            foreach_point_neighbor(coords, coords, nhs,
                                   parallelization_backend = PolyesterBackend()) do point,
                                                                                    neighbor,
                                                                                    pos_diff,
                                                                                    distance
                Threads.atomic_add!(n_pairs, 1)
            end
            @test n_pairs[] == n_pairs_expected

            # Also test a freshly created ("copied") template neighborhood search,
            # as is done when a simulation code needs several `CellListMapNeighborhoodSearch`s.
            template_nhs = CellListMapNeighborhoodSearch(NDIMS; points_equal_neighbors)
            copied_nhs = copy_neighborhood_search(template_nhs, search_radius, n_points)

            initialize!(copied_nhs, coords, coords)

            neighbors_copied = [Int[] for _ in axes(coords, 2)]
            foreach_point_neighbor(coords, coords, copied_nhs,
                                   parallelization_backend = SerialBackend()) do point,
                                                                                 neighbor,
                                                                                 pos_diff,
                                                                                 distance
                push!(neighbors_copied[point], neighbor)
            end
            @test sort.(neighbors_copied) == neighbors_expected
        end
    end

    @testset "Two Different Point Sets" begin
        # `points_equal_neighbors = true` requires `x === y`, so cross-set searches
        # (`x !== y`) must use `points_equal_neighbors = false`.
        search_radius = 0.5
        NDIMS = 2
        coords_x = point_cloud((6, 7), search_radius, seed = 1)
        coords_y = point_cloud((5, 5), search_radius, seed = 3, shuffle = true, sort = false)

        trivial_nhs = TrivialNeighborhoodSearch{NDIMS}(; search_radius,
                                                       eachpoint = axes(coords_y, 2))
        neighbors_expected = [Int[] for _ in axes(coords_x, 2)]
        foreach_point_neighbor(coords_x, coords_y, trivial_nhs,
                               parallelization_backend = SerialBackend()) do point, neighbor,
                                                                              pos_diff, distance
            push!(neighbors_expected[point], neighbor)
        end

        nhs = CellListMapNeighborhoodSearch(NDIMS; search_radius,
                                            points_equal_neighbors = false)
        initialize!(nhs, coords_x, coords_y)

        neighbors = [Int[] for _ in axes(coords_x, 2)]
        foreach_point_neighbor(coords_x, coords_y, nhs,
                               parallelization_backend = SerialBackend()) do point, neighbor,
                                                                              pos_diff, distance
            push!(neighbors[point], neighbor)
        end

        @test sort.(neighbors) == sort.(neighbors_expected)
    end

    @testset "Boundary Case (distance == search_radius)" begin
        # Hand-built example so that the two points are exactly `search_radius` apart,
        # to make sure the `<=` convention matches `TrivialNeighborhoodSearch`.
        search_radius = 1.0
        coords = [0.0 1.0; 0.0 0.0]

        trivial_nhs = TrivialNeighborhoodSearch{2}(; search_radius, eachpoint = 1:2)
        nhs = CellListMapNeighborhoodSearch(2; search_radius, points_equal_neighbors = true)
        initialize!(nhs, coords, coords)

        neighbors_trivial = [Int[] for _ in 1:2]
        foreach_point_neighbor(coords, coords, trivial_nhs,
                               parallelization_backend = SerialBackend()) do point, neighbor,
                                                                              pos_diff, distance
            push!(neighbors_trivial[point], neighbor)
        end

        neighbors = [Int[] for _ in 1:2]
        foreach_point_neighbor(coords, coords, nhs,
                               parallelization_backend = SerialBackend()) do point, neighbor,
                                                                              pos_diff, distance
            push!(neighbors[point], neighbor)
        end

        @test sort.(neighbors) == sort.(neighbors_trivial)
    end
end
