@testset verbose=true "CombinedGridNeighborhoodSearch" begin
    # `f` returns a nested tuple to test the flattening of the results.
    function f(point, neighbor, pos_diff, distance)
        return (pos_diff * distance, (neighbor, 1.0f0))
    end
    op(a, b) = (a[1] + b[1], (a[2][1] + b[2][1], a[2][2] + b[2][2]))

    function neighbor_sums(foreach_point_, system_coords, neighbor_coords, nhs, backend)
        init = (zero(SVector{ndims(nhs), Float64}), (0, 0.0f0))
        results = fill(init, size(system_coords, 2))
        foreach_point_(system_coords, neighbor_coords, nhs;
                       parallelization_backend = backend) do point, neighbors
            results[point] = mapreduce_neighbor(f, op, neighbors; init)
        end

        return results
    end

    # Reference results with a separate `GridNeighborhoodSearch`.
    function neighbor_sums_expected(system_coords, neighbor_coords, cell_list,
                                    search_radius)
        cell_list_ = PointNeighbors.copy_cell_list(cell_list, search_radius, nothing)
        nhs_reference = GridNeighborhoodSearch{3}(; search_radius,
                                                  n_points = size(neighbor_coords, 2),
                                                  cell_list = cell_list_)
        initialize!(nhs_reference, system_coords, neighbor_coords)

        return neighbor_sums(foreach_point, system_coords, neighbor_coords,
                             nhs_reference, SerialBackend())
    end

    search_radius = 2.5

    # Three point sets in the same domain:
    # a perturbed grid, random points, and only a few points.
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

        @testset "$(nameof(foreach_point_)), $(nameof(typeof(backend)))" for foreach_point_ in
                                                                             (foreach_point,
                                                                              foreach_point_unsafe),
                                                                             backend in
                                                                             (SerialBackend(),
                                                                              PolyesterBackend())

            results = neighbor_sums(foreach_point_, system_coords, neighbor_coords, pair,
                                    backend)

            for point in axes(system_coords, 2)
                @test results[point][1] ≈ results_expected[point][1]
                @test results[point][2] == results_expected[point][2]
            end
        end
    end

    @testset "Update" begin
        # Move the points of the second point set.
        coords2_new = coords2 .+ 0.3 .* randn(MersenneTwister(4), size(coords2))
        update!(nhs, 2, coords2_new)

        results_expected = neighbor_sums_expected(coords1, coords2_new, cell_list,
                                                  search_radius)
        results = neighbor_sums(foreach_point, coords1, coords2_new,
                                point_set_pair(nhs, 1, 2), PolyesterBackend())
        @test [result[2] for result in results] ==
              [result[2] for result in results_expected]

        # The cell lists can only be initialized or updated for each point set.
        pair = point_set_pair(nhs, 2, 1)
        @test_throws ArgumentError initialize!(pair, coords2, coords1)
        @test_throws ArgumentError update!(pair, coords2, coords1)
    end

    @testset "Copy" begin
        pair = point_set_pair(nhs, 1, 2)
        @test copy_neighborhood_search(pair, 1.0, 10) isa GridNeighborhoodSearch
        @test requires_update(pair) == (false, true)
        @test ndims(pair) == 3
        @test PointNeighbors.search_radius(pair) == search_radius

        pair2 = @trixi_test_nowarn PointNeighbors.Adapt.adapt_structure(Array, pair)
        @test pair2.system_index == 1
        @test pair2.neighbor_index == 2

        nhs2 = @trixi_test_nowarn PointNeighbors.Adapt.adapt_structure(Array, nhs)
        @test length(nhs2.searches) == 3
    end
end;
