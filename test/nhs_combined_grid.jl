@testset verbose=true "CombinedGridNeighborhoodSearch" begin
    f(point, neighbor, pos_diff, distance) = neighbor

    function neighbor_sums(foreach_point_, system_coords, neighbor_coords, nhs)
        results = zeros(Int, size(system_coords, 2))
        foreach_point_(system_coords, neighbor_coords, nhs;
                       parallelization_backend = SerialBackend()) do point, neighbors
            results[point] = mapreduce_neighbor(f, +, neighbors; init = 0)
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

        return neighbor_sums(foreach_point, system_coords, neighbor_coords, nhs_reference)
    end

    search_radius = 2.5

    # Three point sets in the same domain:
    # a perturbed grid, random points, and only a few points.
    coords1 = point_cloud((4, 8, 9), search_radius, shuffle = true)
    coords2 = 1 .+ 11 * rand(Random.Xoshiro(1), 3, 70)
    coords3 = 1 .+ 11 * rand(Random.Xoshiro(2), 3, 5)
    coords = [coords1, coords2, coords3]

    cell_list = FullGridCellList(; min_corner = minimum(reduce(hcat, coords), dims = 2),
                                 max_corner = maximum(reduce(hcat, coords), dims = 2),
                                 search_radius)
    template = GridNeighborhoodSearch{3}(; cell_list)
    nhs = CombinedGridNeighborhoodSearch(template, search_radius, size.(coords, 2))

    for (point_set, coords_) in enumerate(coords)
        initialize!(nhs, point_set, coords_)
    end

    @testset "Point Set Pair ($system_index, $neighbor_index)" for system_index in 1:3,
                                                                   neighbor_index in 1:3
        system_coords = coords[system_index]
        neighbor_coords = coords[neighbor_index]
        pair = grid_nhs_pair(nhs, system_index, neighbor_index)

        results_expected = neighbor_sums_expected(system_coords, neighbor_coords, cell_list,
                                                  search_radius)

        @testset "$(nameof(foreach_point_))" for foreach_point_ in (foreach_point,
                                                                    foreach_point_unsafe)

            results = neighbor_sums(foreach_point_, system_coords, neighbor_coords, pair)

            @test results == results_expected
        end
    end

    @testset "Update" begin
        # Move the points of the second point set.
        coords2_new = coords2 .+ 0.3 .* randn(Random.Xoshiro(4), size(coords2))
        update!(nhs, 2, coords2_new)

        results_expected = neighbor_sums_expected(coords1, coords2_new, cell_list,
                                                  search_radius)
        results = neighbor_sums(foreach_point, coords1, coords2_new,
                                grid_nhs_pair(nhs, 1, 2))
        @test results == results_expected

        # The pair is a view of the cell lists of the combined search.
        pair = grid_nhs_pair(nhs, 1, 2)
        @test pair.system_search === nhs.searches[1]
        @test pair.neighbor_search === nhs.searches[2]

        # The cell lists can only be initialized or updated for each point set.
        pair = grid_nhs_pair(nhs, 2, 1)
        @test_throws ArgumentError initialize!(pair, coords2, coords1)
        @test_throws ArgumentError update!(pair, coords2, coords1)
    end

    # @testset "Copy" begin
    #     pair = grid_nhs_pair(nhs, 1, 2)
    #     @test copy_neighborhood_search(pair, 1.0, 10) isa GridNeighborhoodSearch
    #     @test requires_update(pair) == (false, true)
    #     @test ndims(pair) == 3
    #     @test PointNeighbors.search_radius(pair) == search_radius

    #     pair2 = @trixi_test_nowarn PointNeighbors.Adapt.adapt_structure(Array, pair)
    #     @test pair2 isa PointNeighbors.GridNHSPair
    #     @test pair2.system_search.cell_list.cells == pair.system_search.cell_list.cells
    #     @test pair2.neighbor_search.cell_list.cells == pair.neighbor_search.cell_list.cells

    #     nhs2 = @trixi_test_nowarn PointNeighbors.Adapt.adapt_structure(Array, nhs)
    #     @test length(nhs2.searches) == 3
    # end
end;
