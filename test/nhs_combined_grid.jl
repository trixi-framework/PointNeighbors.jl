@testset verbose=true "CombinedGridNeighborhoodSearch" begin
    # `f` returns a nested tuple to test the flattening of the results.
    function f(point, neighbor, pos_diff, distance)
        return (pos_diff * distance, (neighbor, 1.0f0))
    end
    op(a, b) = (a[1] + b[1], (a[2][1] + b[2][1], a[2][2] + b[2][2]))

    function neighbor_sums(foreach_point_, system_coords, neighbor_coords, nhs, backend,
                           simd; points = axes(system_coords, 2))
        init = (zero(SVector{ndims(nhs), Float64}), (0, 0.0f0))
        results = fill(init, size(system_coords, 2))
        is_blocked = fill(false, size(system_coords, 2))
        foreach_point_(system_coords, neighbor_coords, nhs; simd, points,
                       parallelization_backend = backend) do point, neighbors
            results[point] = mapreduce_neighbor(f, op, neighbors; init)
            is_blocked[point] = neighbors isa PointNeighbors.NeighborBlocksOfPoint
        end

        return results, is_blocked
    end

    # Reference results with a separate `GridNeighborhoodSearch` and the regular loop.
    function neighbor_sums_expected(system_coords, neighbor_coords, cell_list,
                                    search_radius)
        cell_list_ = PointNeighbors.copy_cell_list(cell_list, search_radius, nothing)
        nhs_reference = GridNeighborhoodSearch{3}(; search_radius,
                                                  n_points = size(neighbor_coords, 2),
                                                  cell_list = cell_list_)
        initialize!(nhs_reference, system_coords, neighbor_coords)

        results,
        _ = neighbor_sums(foreach_point, system_coords, neighbor_coords,
                          nhs_reference, SerialBackend(), Val(false))
        return results
    end

    search_radius = 2.5

    # Three point sets in the same domain:
    # a perturbed grid, random points, and fewer points than a SIMD block.
    coords1 = point_cloud((10, 11, 12), search_radius, shuffle = true)
    coords2 = 1 .+ 11 * rand(MersenneTwister(1), 3, 1000)
    coords3 = 1 .+ 11 * rand(MersenneTwister(2), 3, 5)
    coords = [coords1, coords2, coords3]

    cell_list = FullGridCellList(; min_corner = minimum(reduce(hcat, coords), dims = 2),
                                 max_corner = maximum(reduce(hcat, coords), dims = 2),
                                 search_radius, max_points_per_cell = 200)
    template = GridNeighborhoodSearch{3}(; cell_list, update_strategy = ParallelUpdate())
    nhs = CombinedGridNeighborhoodSearch(template, search_radius, size.(coords, 2))

    for (point_set, coords_) in enumerate(coords)
        initialize!(nhs, point_set, coords_)
    end

    @testset "Point Set Pair ($system_index, $neighbor_index)" for system_index in 1:3,
                                                                   neighbor_index in 1:3
        system_coords = coords[system_index]
        neighbor_coords = coords[neighbor_index]
        pair = point_set_pair(nhs, system_index, neighbor_index)

        results_expected = neighbor_sums_expected(system_coords, neighbor_coords, cell_list,
                                                  search_radius)

        @testset "$(nameof(foreach_point_)), $(nameof(typeof(backend))), simd = $simd" for foreach_point_ in
                                                                                           (foreach_point,
                                                                                            foreach_point_unsafe),
                                                                                           backend in
                                                                                           (SerialBackend(),
                                                                                            PolyesterBackend(),
                                                                                            ThreadsDynamicBackend()),
                                                                                           simd in
                                                                                           (Val(false),
                                                                                            Val(true))

            results,
            is_blocked = neighbor_sums(foreach_point_, system_coords, neighbor_coords,
                                       pair, backend, simd)
            @test all(is_blocked) == (simd isa Val{true})

            for point in axes(system_coords, 2)
                @test results[point][1] ≈ results_expected[point][1]
                @test results[point][2] == results_expected[point][2]
            end
        end
    end

    @testset "Update" begin
        # Move the points of the first two point sets.
        coords1_new = coords1 .+ 0.3 .* randn(MersenneTwister(3), size(coords1))
        coords2_new = coords2 .+ 0.3 .* randn(MersenneTwister(4), size(coords2))
        update!(nhs, 1, coords1_new)
        update!(nhs, 2, coords2_new)

        results_expected = neighbor_sums_expected(coords1_new, coords2_new, cell_list,
                                                  search_radius)
        results,
        is_blocked = neighbor_sums(foreach_point, coords1_new, coords2_new,
                                   point_set_pair(nhs, 1, 2), PolyesterBackend(),
                                   Val(true))
        @test all(is_blocked)
        @test [result[2] for result in results] ==
              [result[2] for result in results_expected]

        # The cell lists can only be initialized or updated for each point set.
        pair = point_set_pair(nhs, 2, 1)
        @test_throws ArgumentError initialize!(pair, coords2, coords1)
        @test_throws ArgumentError update!(pair, coords2, coords1)
    end

    @testset "Inactive Points" begin
        # Some points of the first point set are inactive, so they are neither in its
        # cell list nor in `points`, like inactive particles of a buffer.
        active = setdiff(axes(coords1, 2), [3, 17, 200, 1000])
        points = view(active, :)
        initialize!(nhs, 1, coords1, eachindex = points)
        initialize!(nhs, 2, coords2)

        results_expected = neighbor_sums_expected(coords1, coords2, cell_list,
                                                  search_radius)
        results,
        is_blocked = neighbor_sums(foreach_point, coords1, coords2,
                                   point_set_pair(nhs, 1, 2), PolyesterBackend(),
                                   Val(true); points)
        @test all(is_blocked[points])
        @test count(is_blocked) == length(points)
        @test [result[1] for result in results[points]] ≈
              [result[1] for result in results_expected[points]]
        @test [result[2] for result in results[points]] ==
              [result[2] for result in results_expected[points]]
    end

    @testset "Points Not in the Cell List" begin
        # With `simd = Val(true)`, all `points` must be in the cell list of the system
        # point set.
        initialize!(nhs, 1, coords1, eachindex = 1:2:size(coords1, 2))
        initialize!(nhs, 2, coords2)

        @test_throws "all `points` must be in the cell list" neighbor_sums(foreach_point,
                                                                           coords1,
                                                                           coords2,
                                                                           point_set_pair(nhs,
                                                                                          1,
                                                                                          2),
                                                                           PolyesterBackend(),
                                                                           Val(true))
    end

    @testset "Different Element Types" begin
        # With `simd = Val(true)`, all coordinates must have the same type.
        initialize!(nhs, 1, coords1)
        initialize!(nhs, 2, coords2)
        @test_throws "must have the same element type" neighbor_sums(foreach_point,
                                                                     coords1,
                                                                     Float32.(coords2),
                                                                     point_set_pair(nhs,
                                                                                    1,
                                                                                    2),
                                                                     SerialBackend(),
                                                                     Val(true))
    end

    @testset "Copy" begin
        pair = point_set_pair(nhs, 1, 2)
        @test copy_neighborhood_search(pair, 1.0, 10) isa GridNeighborhoodSearch
        @test requires_update(pair) == (true, true)
        @test ndims(pair) == 3
        @test PointNeighbors.search_radius(pair) == search_radius

        pair2 = @trixi_test_nowarn PointNeighbors.Adapt.adapt_structure(Array, pair)
        @test pair2.system_index == 1
        @test pair2.neighbor_index == 2

        nhs2 = @trixi_test_nowarn PointNeighbors.Adapt.adapt_structure(Array, nhs)
        @test length(nhs2.searches) == 3
    end
end;

@testset verbose=true "CombinedGridNeighborhoodSearch With a Single Point Set" begin
    # `f` returns a nested tuple to test the flattening of the results.
    function f(point, neighbor, pos_diff, distance)
        return (pos_diff * distance, (neighbor, 1.0f0))
    end
    op(a, b) = (a[1] + b[1], (a[2][1] + b[2][1], a[2][2] + b[2][2]))

    function neighbor_sums_single(foreach_point_, coords, nhs, backend, simd;
                           system_coords = coords, points = axes(system_coords, 2))
        init = (zero(SVector{ndims(nhs), Float64}), (0, 0.0f0))
        results = fill(init, size(system_coords, 2))
        is_blocked = fill(false, size(system_coords, 2))
        foreach_point_(system_coords, coords, nhs; simd, points,
                       parallelization_backend = backend) do point, neighbors
            results[point] = mapreduce_neighbor(f, op, neighbors; init)
            is_blocked[point] = neighbors isa PointNeighbors.NeighborBlocksOfPoint
        end

        return results, is_blocked
    end

    # The last cloud has up to 125 points per cell,
    # which requires multiple groups of blocks in the filter.
    cases = [((30, 31), 2.5, Float64), ((10, 11, 12), 2.5, Float64),
        ((10, 11, 12), 2.5, Float32), ((20, 21, 22), 4.5, Float64)]

    @testset verbose=true "$(length(cloud_size))D, $ELTYPE, search radius $search_radius" for (cloud_size,
                                                                                               search_radius,
                                                                                               ELTYPE) in
                                                                                              cases

        NDIMS = length(cloud_size)
        coords = convert.(ELTYPE,
                          point_cloud(cloud_size, search_radius, shuffle = true))
        n_points = size(coords, 2)
        cell_list = FullGridCellList(; min_corner = minimum(coords, dims = 2),
                                     max_corner = maximum(coords, dims = 2),
                                     search_radius, max_points_per_cell = 200)
        template = GridNeighborhoodSearch{NDIMS}(; cell_list,
                                                 update_strategy = ParallelUpdate())
        nhs_combined = CombinedGridNeighborhoodSearch(template, search_radius,
                                                      (n_points,))
        initialize!(nhs_combined, 1, coords)
        nhs = point_set_pair(nhs_combined, 1, 1)

        results_expected,
        _ = neighbor_sums_single(foreach_point, coords, nhs,
                          SerialBackend(), Val(false))

        @testset "$(nameof(foreach_point_)), $(nameof(typeof(backend)))" for foreach_point_ in
                                                                             (foreach_point,
                                                                              foreach_point_unsafe),
                                                                             backend in
                                                                             (SerialBackend(),
                                                                              PolyesterBackend(),
                                                                              ThreadsDynamicBackend())

            results,
            is_blocked = neighbor_sums_single(foreach_point_, coords, nhs, backend,
                                       Val(true))
            @test all(is_blocked)

            for point in axes(coords, 2)
                @test results[point][1] ≈ results_expected[point][1]
                @test results[point][2] == results_expected[point][2]
            end
        end

        @testset "Subset of the Points ($(nameof(typeof(points))))" for points in
                                                                        (2:(n_points - 3),
                                                                         1:2:n_points,
                                                                         view(randperm(MersenneTwister(1),
                                                                                       n_points),
                                                                              1:(n_points - 5)))
            results,
            is_blocked = neighbor_sums_single(foreach_point, coords, nhs,
                                       PolyesterBackend(), Val(true); points)

            # The other points are not computed.
            @test all(is_blocked[points])
            @test count(is_blocked) == length(points)
            @test [result[1] for result in results[points]] ≈
                  [result[1] for result in results_expected[points]]
            @test [result[2] for result in results[points]] ==
                  [result[2] for result in results_expected[points]]
        end

        @testset "Points Not in the Cell List" begin
            # With `simd = Val(true)`, all `points` must be in the cell list.
            nhs_subset = deepcopy(nhs_combined)
            initialize!(nhs_subset, 1, coords, eachindex = 1:2:n_points)
            @test_throws "all `points` must be in the cell list" neighbor_sums_single(foreach_point,
                                                                               coords,
                                                                               point_set_pair(nhs_subset,
                                                                                              1,
                                                                                              1),
                                                                               PolyesterBackend(),
                                                                               Val(true))
        end
    end

    @testset "Mixed Precision" begin
        # `Float64` coordinates with a `Float32` search radius.
        search_radius = 2.5f0
        coords = point_cloud((10, 11, 12), search_radius, shuffle = true)
        n_points = size(coords, 2)
        cell_list = FullGridCellList(; min_corner = minimum(coords, dims = 2),
                                     max_corner = maximum(coords, dims = 2),
                                     search_radius)
        template = GridNeighborhoodSearch{3}(; cell_list,
                                             update_strategy = ParallelUpdate())
        nhs_combined = CombinedGridNeighborhoodSearch(template, search_radius,
                                                      (n_points,))
        initialize!(nhs_combined, 1, coords)
        nhs = point_set_pair(nhs_combined, 1, 1)

        function count_neighbors(simd)
            n_neighbors = zeros(Int, n_points)
            foreach_point(coords, coords, nhs; simd) do point, neighbors
                n_neighbors[point] = mapreduce_neighbor((point, neighbor, pos_diff,
                                                         distance) -> 1, +, neighbors;
                                                        init = 0)
            end
            return n_neighbors
        end

        @test count_neighbors(Val(true)) == count_neighbors(Val(false))
    end
end;
