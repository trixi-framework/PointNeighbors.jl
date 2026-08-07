# This file contains tests for the generic functions in `src/neighborhood_search.jl` and
# tests comparing all NHS implementations against the `TrivialNeighborhoodSearch`.
@testset verbose=true "All Neighborhood Searches" begin
    @testset "periodicity rounding errors" begin
        for T in (Float32, Float64)
            box = PeriodicBox(; min_corner = SVector(T(0), T(0)),
                              max_corner = SVector(T(1), T(1)))

            coords = (SVector(prevfloat(T(0)), T(0.5)),
                      SVector(T(0), T(0.5)),
                      SVector(nextfloat(T(0)), T(0.5)),
                      SVector(prevfloat(T(1)), T(0.5)),
                      SVector(T(1), T(0.5)),
                      SVector(nextfloat(T(1)), T(0.5)),
                      SVector(T(0.5), prevfloat(T(0))),
                      SVector(T(0.5), T(0)),
                      SVector(T(0.5), nextfloat(T(0))),
                      SVector(T(0.5), prevfloat(T(1))),
                      SVector(T(0.5), T(1)),
                      SVector(T(0.5), nextfloat(T(1))))

            # Test `periodic_coords`.
            for x in coords
                xp = PointNeighbors.periodic_coords(x, box)

                # Test that the periodic coordinates are within the periodic box.
                @test all(box.min_corner .<= xp)
                @test all(xp .<= box.max_corner)

                # Test that the periodic coordinates are equivalent to
                # the original coordinates up to periodicity.
                @test xp[1] in (x[1], x[1] - box.size[1], x[1] + box.size[1])
                @test xp[2] in (x[2], x[2] - box.size[2], x[2] + box.size[2])
            end

            # Test that `cell_coords`, which is using integer modulo arithmetic instead of
            # `periodic_coords`, handles rounding errors at periodic boundaries.
            search_radius = T(0.1)

            cell_list = FullGridCellList(; min_corner = box.min_corner,
                                         max_corner = box.max_corner,
                                         search_radius)

            nhs = GridNeighborhoodSearch{2}(; search_radius,
                                            n_points = length(coords),
                                            periodic_box = box, cell_list)

            for x in coords
                cell = PointNeighbors.cell_coords(x, nhs)
                periodic_coords_ = PointNeighbors.periodic_coords(x, box)
                periodic_cell = PointNeighbors.cell_coords(periodic_coords_, nhs)

                @test cell == periodic_cell
                @test all(2 <= cell[i] <= nhs.n_cells[i] + 1 for i in eachindex(cell))
            end
        end
    end

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
                GridNeighborhoodSearch{NDIMS}(; search_radius, n_points,
                                              periodic_box = periodic_boxes[i],
                                              cell_list = FullGridCellList(; min_corner,
                                                                           max_corner,
                                                                           search_radius,
                                                                           backend = PointNeighbors.CompactVectorOfVectors{Int32})),
                PrecomputedNeighborhoodSearch{NDIMS}(; search_radius, n_points,
                                                     periodic_box = periodic_boxes[i]),
                PrecomputedNeighborhoodSearch{NDIMS}(; search_radius, n_points,
                                                     periodic_box = periodic_boxes[i],
                                                     backend = Vector{Vector{Int32}}),
                GridNeighborhoodSearch{NDIMS}(; search_radius, n_points,
                                              periodic_box = periodic_boxes[i],
                                              cell_list = SpatialHashingCellList{NDIMS}(list_size = 2 *
                                                                                                    n_points))
            ]

            names = [
                "`TrivialNeighborhoodSearch`",
                "`GridNeighborhoodSearch`",
                "`GridNeighborhoodSearch` with `FullGridCellList` with `DynamicVectorOfVectors`",
                "`GridNeighborhoodSearch` with `FullGridCellList` with `Vector{Vector}`",
                "`GridNeighborhoodSearch` with `FullGridCellList` with `CompactVectorOfVectors`",
                "`PrecomputedNeighborhoodSearch`",
                "`PrecomputedNeighborhoodSearch` with `Vector{Vector}`",
                "`GridNeighborhoodSearch` with `SpatialHashingCellList`"
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
                GridNeighborhoodSearch{NDIMS}(periodic_box = periodic_boxes[i],
                                              cell_list = FullGridCellList(min_corner = periodic_boxes[i].min_corner,
                                                                           max_corner = periodic_boxes[i].max_corner,
                                                                           backend = PointNeighbors.CompactVectorOfVectors{Int32})),
                PrecomputedNeighborhoodSearch{NDIMS}(periodic_box = periodic_boxes[i]),
                PrecomputedNeighborhoodSearch{NDIMS}(periodic_box = periodic_boxes[i],
                                                     backend = Vector{Vector{Int32}}),
                GridNeighborhoodSearch{NDIMS}(periodic_box = periodic_boxes[i],
                                              cell_list = SpatialHashingCellList{NDIMS}(list_size = 2 *
                                                                                                    n_points))
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
                    push!(neighbors[point], neighbor)
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
            search_radius = 2.5
            coords = point_cloud(cloud_size, search_radius, seed = seed)
            NDIMS = length(cloud_size)
            n_points = size(coords, 2)

            # Use different coordinates for `initialize!` and then `update!` with the
            # correct coordinates to make sure that `update!` is working as well.
            coords_initialize = point_cloud(cloud_size, search_radius, seed = 1)

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
                push!(neighbors_expected[point], neighbor)
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
                GridNeighborhoodSearch{NDIMS}(; search_radius, n_points,
                                              cell_list = FullGridCellList(; min_corner,
                                                                           max_corner,
                                                                           search_radius,
                                                                           backend = PointNeighbors.CompactVectorOfVectors{Int32})),
                PrecomputedNeighborhoodSearch{NDIMS}(; search_radius, n_points),
                PrecomputedNeighborhoodSearch{NDIMS}(; search_radius, n_points,
                                                     backend = Vector{Vector{Int}}),
                GridNeighborhoodSearch{NDIMS}(; search_radius, n_points,
                                              cell_list = SpatialHashingCellList{NDIMS}(list_size = 2 *
                                                                                                    n_points)),
                GridNeighborhoodSearch{NDIMS}(; search_radius, n_points,
                                              cell_list = SpatialHashingCellList{NDIMS}(list_size = 2 *
                                                                                                    n_points,
                                                                                        backend = Vector{Vector{Int32}}))
            ]

            names = [
                "`GridNeighborhoodSearch` with `SemiParallelUpdate`",
                "`GridNeighborhoodSearch` with `SerialIncrementalUpdate`",
                "`GridNeighborhoodSearch` with `SerialUpdate`",
                "`GridNeighborhoodSearch` with `FullGridCellList` with `DynamicVectorOfVectors` and `ParallelUpdate`",
                "`GridNeighborhoodSearch` with `FullGridCellList` with `DynamicVectorOfVectors` and `ParallelIncrementalUpdate`",
                "`GridNeighborhoodSearch` with `FullGridCellList` with `DynamicVectorOfVectors` and `SemiParallelUpdate`",
                "`GridNeighborhoodSearch` with `FullGridCellList` with `Vector{Vector}`",
                "`GridNeighborhoodSearch` with `FullGridCellList` with `CompactVectorOfVectors`",
                "`PrecomputedNeighborhoodSearch`",
                "`PrecomputedNeighborhoodSearch` with `Vector{Vector}`",
                "`GridNeighborhoodSearch` with `SpatialHashingCellList` with `DynamicVectorOfVectors`",
                "`GridNeighborhoodSearch` with `SpatialHashingCellList` with `Vector{Vector}`"
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
                GridNeighborhoodSearch{NDIMS}(cell_list = FullGridCellList(; min_corner,
                                                                           max_corner,
                                                                           backend = PointNeighbors.CompactVectorOfVectors{Int32})),
                PrecomputedNeighborhoodSearch{NDIMS}(),
                PrecomputedNeighborhoodSearch{NDIMS}(backend = Vector{Vector{Int32}}),
                GridNeighborhoodSearch{NDIMS}(cell_list = SpatialHashingCellList{NDIMS}(list_size = 2 *
                                                                                                    n_points)),
                GridNeighborhoodSearch{NDIMS}(cell_list = SpatialHashingCellList{NDIMS}(list_size = 2 *
                                                                                                    n_points,
                                                                                        backend = Vector{Vector{Int32}}))
            ]
            copied_nhs = copy_neighborhood_search.(template_nhs, search_radius, n_points)
            append!(neighborhood_searches, copied_nhs)

            names_copied = [name * " copied" for name in names]
            append!(names, names_copied)

            @testset verbose=true "$(names[i])" for i in eachindex(names)
                nhs = neighborhood_searches[i]

                # Initialize with `seed = 1`
                initialize!(nhs, coords_initialize, coords_initialize)

                # For other seeds, update with the correct coordinates.
                # This way, we test only `initialize!` when `seed == 1`,
                # and `initialize!` plus `update!` else.
                if seed != 1
                    update!(nhs, coords, coords)
                end

                # Test the regular `foreach_point_neighbor`
                @testset "`foreach_point_neighbor`" begin
                    neighbors = [Int[] for _ in axes(coords, 2)]
                    foreach_point_neighbor(coords, coords, nhs,
                                           parallelization_backend = SerialBackend()) do point,
                                                                                         neighbor,
                                                                                         pos_diff,
                                                                                         distance
                        push!(neighbors[point], neighbor)
                    end

                    @test sort.(neighbors) == neighbors_expected
                end

                # Test manual loop with `foreach_neighbor`
                @testset "Manual Loop with `foreach_neighbor`" begin
                    neighbors_manual = [Int[] for _ in axes(coords, 2)]
                    for point in axes(coords, 2)
                        foreach_neighbor(coords, coords, nhs,
                                         point) do point, neighbor, pos_diff, distance
                            push!(neighbors_manual[point], neighbor)
                        end
                    end

                    @test sort.(neighbors_manual) == neighbors_expected

                    # Test that `foreach_neighbor` does not allocate.
                    point = first(axes(coords, 2))
                    function allocations_empty_foreach_neighbor(coords, nhs, point)
                        @allocated(foreach_neighbor((point, neighbor, pos_diff,
                                                     distance) -> nothing,
                                                    coords, coords, nhs, point))
                    end
                    @test allocations_empty_foreach_neighbor(coords, nhs, point) == 0
                end

                # Repeat with foreach_point_neighbor_unsafe
                @testset "`foreach_point_neighbor_unsafe`" begin
                    neighbors_unsafe = [Int[] for _ in axes(coords, 2)]
                    foreach_point_neighbor_unsafe(coords, coords, nhs,
                                                  parallelization_backend = SerialBackend()) do point,
                                                                                                neighbor,
                                                                                                pos_diff,
                                                                                                distance
                        push!(neighbors_unsafe[point], neighbor)
                    end

                    @test sort.(neighbors_unsafe) == neighbors_expected
                end

                # Repeat with manual loop with `foreach_neighbor_unsafe`
                @testset "Manual Loop with `foreach_neighbor_unsafe`" begin
                    neighbors_manual_unsafe = [Int[] for _ in axes(coords, 2)]
                    for point in axes(coords, 2)
                        foreach_neighbor_unsafe(coords, coords, nhs,
                                                point) do point, neighbor,
                                                          pos_diff, distance
                            push!(neighbors_manual_unsafe[point], neighbor)
                        end
                    end

                    @test sort.(neighbors_manual_unsafe) == neighbors_expected
                end

                @testset "`mapreduce_neighbor`" begin
                    neighbor_sums = map(axes(coords, 2)) do point
                        mapreduce_neighbor(+, coords, coords, nhs, point;
                                           init = 0) do point_, neighbor,
                                                        pos_diff, distance
                            point_ == point || error("incorrect point index")
                            neighbor
                        end
                    end

                    @test neighbor_sums == sum.(neighbors_expected)

                    # Test that `mapreduce_neighbor` does not allocate.
                    point = first(axes(coords, 2))
                    function allocations_count_neighbors(coords, nhs, point)
                        @allocated(mapreduce_neighbor((point, neighbor, pos_diff,
                                                       distance) -> neighbor,
                                                      +, coords, coords, nhs, point;
                                                      init = 0))
                    end
                    @test allocations_count_neighbors(coords, nhs, point) == 0

                    @test_throws UndefKeywordError mapreduce_neighbor(+, coords, coords,
                                                                      nhs,
                                                                      first(axes(coords,
                                                                                 2))) do point_,
                                                                                         neighbor,
                                                                                         pos_diff,
                                                                                         distance
                        neighbor
                    end
                end

                @testset "`mapreduce_neighbor_unsafe`" begin
                    neighbor_sums = map(axes(coords, 2)) do point
                        mapreduce_neighbor_unsafe(+, coords, coords, nhs, point;
                                                  init = 0) do point_, neighbor,
                                                               pos_diff, distance
                            point_ == point || error("incorrect point index")
                            neighbor
                        end
                    end

                    @test neighbor_sums == sum.(neighbors_expected)

                    @test_throws UndefKeywordError mapreduce_neighbor_unsafe(+,
                                                                             coords, coords,
                                                                             nhs,
                                                                             first(axes(coords,
                                                                                        2))) do point_,
                                                                                                neighbor,
                                                                                                pos_diff,
                                                                                                distance
                        neighbor
                    end

                    # Test the reduction over an empty neighborhood.
                    empty_nhs = copy_neighborhood_search(nhs, search_radius,
                                                         size(coords, 2))

                    # Initialize the NHS with an empty set of neighbors.
                    empty_coords = similar(coords, size(coords, 1), 0)
                    initialize!(empty_nhs, coords, empty_coords)
                    point = first(axes(coords, 2))

                    function f(point, neighbor, pos_diff, distance)
                        error("`f` must not be called for an empty neighborhood")
                    end
                    function op(a, b)
                        error("`op` must not be called for an empty neighborhood")
                    end

                    # Using a non-neutral `init` here is intentional: for an empty
                    # neighborhood, `init` must be returned unchanged.
                    result = mapreduce_neighbor_unsafe(f, op, coords, empty_coords,
                                                       empty_nhs, point; init = 123)
                    @test result == 123
                end
            end
        end
    end
end;
