# This file contains tests for the generic functions in `src/neighborhood_search.jl` and
# tests comparing all NHS implementations against the `TrivialNeighborhoodSearch`.
@testset verbose=true "All Neighborhood Searches" begin
    @testset verbose=true "Periodicity" begin
        # These examples are constructed by hand and are therefore a good test for the
        # trivial neighborhood search as well.
        # (As opposed to the tests below that are just comparing against the trivial NHS.)

        # Names, coordinates and corresponding periodic boxes for each test
        names = [
            "Simple Example 2D",
            "Box Not Multiple of Search Radius 2D",
            "Simple Example 3D"
        ]

        coordinates = [
            [-0.08 0.0 0.18 0.1 -0.08
             -0.12 -0.05 -0.09 0.15 0.39],
            [-0.08 0.0 0.18 0.1 -0.08
             -0.12 -0.05 -0.09 0.15 0.42],
            [-0.08 0.0 0.18 0.1 -0.08
             -0.12 -0.05 -0.09 0.15 0.39
             0.14 0.34 0.12 0.06 0.13]
        ]

        periodic_boxes = [
            PeriodicBox(min_corner = [-0.1, -0.2], max_corner = [0.2, 0.4]),
            # The `GridNeighborhoodSearch` is forced to round up the cell sizes in this test
            # to avoid split cells.
            PeriodicBox(min_corner = [-0.1, -0.2], max_corner = [0.205, 0.43]),
            PeriodicBox(min_corner = [-0.1, -0.2, 0.05], max_corner = [0.2, 0.4, 0.35])
        ]

        @testset verbose=true "$(names[i])" for i in eachindex(names)
            coords = coordinates[i]

            NDIMS = size(coords, 1)
            n_points = size(coords, 2)
            search_radius = 0.1

            min_corner = periodic_boxes[i].min_corner
            max_corner = periodic_boxes[i].max_corner

            neighborhood_searches = [
                TrivialNeighborhoodSearch{NDIMS}(; search_radius, eachpoint = 1:n_points,
                                                 periodic_box = periodic_boxes[i]),
                GridNeighborhoodSearch{NDIMS}(; search_radius, n_points,
                                              periodic_box = periodic_boxes[i]),
                GridNeighborhoodSearch{NDIMS}(; search_radius, n_points,
                                              periodic_box = periodic_boxes[i],
                                              cell_list = FullGridCellList(; min_corner,
                                                                           max_corner,
                                                                           search_radius)),
                GridNeighborhoodSearch{NDIMS}(; search_radius, n_points,
                                              periodic_box = periodic_boxes[i],
                                              cell_list = FullGridCellList(; min_corner,
                                                                           max_corner,
                                                                           search_radius,
                                                                           backend = Vector{Vector{Int32}})),
                PrecomputedNeighborhoodSearch{NDIMS}(; search_radius, n_points,
                                                     periodic_box = periodic_boxes[i])
            ]

            names = [
                "`TrivialNeighborhoodSearch`",
                "`GridNeighborhoodSearch`",
                "`GridNeighborhoodSearch` with `FullGridCellList` with `DynamicVectorOfVectors`",
                "`GridNeighborhoodSearch` with `FullGridCellList` with `Vector{Vector}`",
                "`PrecomputedNeighborhoodSearch`"
            ]

            # Also test copied templates
            template_nhs = [
                TrivialNeighborhoodSearch{NDIMS}(periodic_box = periodic_boxes[i]),
                GridNeighborhoodSearch{NDIMS}(periodic_box = periodic_boxes[i]),
                GridNeighborhoodSearch{NDIMS}(periodic_box = periodic_boxes[i],
                                              cell_list = FullGridCellList(min_corner = periodic_boxes[i].min_corner,
                                                                           max_corner = periodic_boxes[i].max_corner)),
                GridNeighborhoodSearch{NDIMS}(periodic_box = periodic_boxes[i],
                                              cell_list = FullGridCellList(min_corner = periodic_boxes[i].min_corner,
                                                                           max_corner = periodic_boxes[i].max_corner,
                                                                           backend = Vector{Vector{Int32}})),
                PrecomputedNeighborhoodSearch{NDIMS}(periodic_box = periodic_boxes[i])
            ]
            copied_nhs = copy_neighborhood_search.(template_nhs, search_radius, n_points)
            append!(neighborhood_searches, copied_nhs)

            names_copied = [name * " copied" for name in names]
            append!(names, names_copied)

            # Run this for every neighborhood search
            @testset "$(names[j])" for j in eachindex(names)
                nhs = neighborhood_searches[j]

                initialize!(nhs, coords, coords)

                neighbors = [Int[] for _ in axes(coords, 2)]

                foreach_point_neighbor(coords, coords, nhs,
                                       points = axes(coords, 2)) do point, neighbor,
                                                                    pos_diff, distance
                    append!(neighbors[point], neighbor)
                end

                # All of these tests are designed to yield the same neighbor lists.
                # Note that we have to sort the neighbor lists because neighborhood searches
                # might produce different orders.
                @test sort(neighbors[1]) == [1, 3, 5]
                @test sort(neighbors[2]) == [2]
                @test sort(neighbors[3]) == [1, 3]
                @test sort(neighbors[4]) == [4]
                @test sort(neighbors[5]) == [1, 5]
            end
        end
    end

    @testset verbose=true "Compare Against `TrivialNeighborhoodSearch`" begin
        cloud_sizes = [
            (10, 11),
            (100, 90),
            (9, 10, 7),
            (39, 40, 41)
        ]

        seeds = [1, 2]
        name(size,
             seed) = "$(length(size))D with $(prod(size)) Particles " *
                     "($(seed == 1 ? "`initialize!`" : "`update!`"))"
        @testset verbose=true "$(name(cloud_size, seed)))" for cloud_size in cloud_sizes,
                                                               seed in seeds
            coords = point_cloud(cloud_size, seed = seed)
            NDIMS = length(cloud_size)
            n_points = size(coords, 2)
            search_radius = 2.5

            # Use different coordinates for `initialize!` and then `update!` with the
            # correct coordinates to make sure that `update!` is working as well.
            coords_initialize = point_cloud(cloud_size, seed = 1)

            # Compute expected neighbor lists by brute-force looping over all points
            # as potential neighbors (`TrivialNeighborhoodSearch`).
            trivial_nhs = TrivialNeighborhoodSearch{NDIMS}(; search_radius,
                                                           eachpoint = axes(coords, 2))

            neighbors_expected = [Int[] for _ in axes(coords, 2)]

            foreach_point_neighbor(coords, coords, trivial_nhs,
                                   parallelization_backend = SerialBackend()) do point,
                                                                                 neighbor,
                                                                                 pos_diff,
                                                                                 distance
                append!(neighbors_expected[point], neighbor)
            end

            # Expand the domain by `search_radius`, as we need the neighboring cells of
            # the minimum and maximum coordinates as well.
            min_corner = minimum(coords, dims = 2) .- search_radius
            max_corner = maximum(coords, dims = 2) .+ search_radius

            neighborhood_searches = [
                GridNeighborhoodSearch{NDIMS}(; search_radius, n_points,
                                              update_strategy = SemiParallelUpdate()),
                GridNeighborhoodSearch{NDIMS}(; search_radius, n_points,
                                              update_strategy = SerialIncrementalUpdate()),
                GridNeighborhoodSearch{NDIMS}(; search_radius, n_points,
                                              update_strategy = SerialUpdate()),
                GridNeighborhoodSearch{NDIMS}(; search_radius, n_points,
                                              cell_list = FullGridCellList(; min_corner,
                                                                           max_corner,
                                                                           search_radius),
                                              update_strategy = ParallelUpdate()),
                GridNeighborhoodSearch{NDIMS}(; search_radius, n_points,
                                              cell_list = FullGridCellList(; min_corner,
                                                                           max_corner,
                                                                           search_radius),
                                              update_strategy = ParallelIncrementalUpdate()),
                GridNeighborhoodSearch{NDIMS}(; search_radius, n_points,
                                              cell_list = FullGridCellList(; min_corner,
                                                                           max_corner,
                                                                           search_radius),
                                              update_strategy = SemiParallelUpdate()),
                GridNeighborhoodSearch{NDIMS}(; search_radius, n_points,
                                              cell_list = FullGridCellList(; min_corner,
                                                                           max_corner,
                                                                           search_radius,
                                                                           backend = Vector{Vector{Int}})),
                PrecomputedNeighborhoodSearch{NDIMS}(; search_radius, n_points)
            ]

            names = [
                "`GridNeighborhoodSearch` with `SemiParallelUpdate`",
                "`GridNeighborhoodSearch` with `SerialIncrementalUpdate`",
                "`GridNeighborhoodSearch` with `SerialUpdate`",
                "`GridNeighborhoodSearch` with `FullGridCellList` with `DynamicVectorOfVectors` and `ParallelUpdate`",
                "`GridNeighborhoodSearch` with `FullGridCellList` with `DynamicVectorOfVectors` and `ParallelIncrementalUpdate`",
                "`GridNeighborhoodSearch` with `FullGridCellList` with `DynamicVectorOfVectors` and `SemiParallelUpdate`",
                "`GridNeighborhoodSearch` with `FullGridCellList` with `Vector{Vector}`",
                "`PrecomputedNeighborhoodSearch`"
            ]

            # Also test copied templates
            template_nhs = [
                GridNeighborhoodSearch{NDIMS}(),
                GridNeighborhoodSearch{NDIMS}(update_strategy = SerialIncrementalUpdate()),
                GridNeighborhoodSearch{NDIMS}(update_strategy = SerialUpdate()),
                GridNeighborhoodSearch{NDIMS}(cell_list = FullGridCellList(; min_corner,
                                                                           max_corner)),
                GridNeighborhoodSearch{NDIMS}(cell_list = FullGridCellList(; min_corner,
                                                                           max_corner),
                                              update_strategy = ParallelIncrementalUpdate()),
                GridNeighborhoodSearch{NDIMS}(cell_list = FullGridCellList(; min_corner,
                                                                           max_corner),
                                              update_strategy = SemiParallelUpdate()),
                GridNeighborhoodSearch{NDIMS}(cell_list = FullGridCellList(; min_corner,
                                                                           max_corner,
                                                                           backend = Vector{Vector{Int32}})),
                PrecomputedNeighborhoodSearch{NDIMS}()
            ]
            copied_nhs = copy_neighborhood_search.(template_nhs, search_radius, n_points)
            append!(neighborhood_searches, copied_nhs)

            names_copied = [name * " copied" for name in names]
            append!(names, names_copied)

            @testset "$(names[i])" for i in eachindex(names)
                nhs = neighborhood_searches[i]

                # Initialize with `seed = 1`
                initialize!(nhs, coords_initialize, coords_initialize)

                # For other seeds, update with the correct coordinates.
                # This way, we test only `initialize!` when `seed == 1`,
                # and `initialize!` plus `update!` else.
                if seed != 1
                    update!(nhs, coords, coords)
                end

                neighbors = [Int[] for _ in axes(coords, 2)]

                foreach_point_neighbor(coords, coords, nhs,
                                       parallelization_backend = SerialBackend()) do point,
                                                                                     neighbor,
                                                                                     pos_diff,
                                                                                     distance
                    append!(neighbors[point], neighbor)
                end

                @test sort.(neighbors) == neighbors_expected
            end
        end
    end
end;
