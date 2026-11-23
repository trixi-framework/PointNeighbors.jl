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

    @testset "Sorting Points" begin
        (; min_corner, max_corner) = cell_list
        template_sorting = SortingGridNeighborhoodSearch{3}(; min_corner, max_corner)
        nhs_sorting = CombinedGridNeighborhoodSearch(template_sorting, search_radius,
                                                     size.(coords, 2))
        initialize!(nhs_sorting, 1, coords1)
        initialize!(nhs_sorting, 2, coords2)

        pair = grid_nhs_pair(nhs_sorting, 1, 2)
        @test sorts_points(pair)
        @test !sorts_points(grid_nhs_pair(nhs, 1, 2))

        # Reorder both point sets with the permutations of their own cell lists
        perm1 = point_permutation(grid_nhs_pair(nhs_sorting, 1, 1))
        perm2 = point_permutation(pair)
        @test perm2 == point_permutation(grid_nhs_pair(nhs_sorting, 2, 2))
        @test sort(perm1) == axes(coords1, 2)
        @test sort(perm2) == axes(coords2, 2)

        # Sums of the original indices of the neighbors
        results_expected = zeros(Int, size(coords1, 2))
        foreach_point_neighbor(coords1, coords2, grid_nhs_pair(nhs, 1, 2),
                               parallelization_backend = SerialBackend()) do point,
                                                                             neighbor,
                                                                             pos_diff,
                                                                             distance
            results_expected[point] += neighbor
        end

        results = zeros(Int, size(coords1, 2))
        foreach_point_neighbor(coords1[:, perm1], coords2[:, perm2],
                               pair,
                               parallelization_backend = SerialBackend()) do point,
                                                                             neighbor,
                                                                             pos_diff,
                                                                             distance
            results[point] += perm2[neighbor]
        end

        # Point `i` of the sorted points is point `perm1[i]` of the original points
        @test results == results_expected[perm1]
    end

    @testset verbose=true "Mixed Precision" begin
        # All neighbors of all points with `pos_diff` and `distance`.
        function neighbor_pairs(foreach_point_neighbor_, system_coords, neighbor_coords,
                                nhs)
            pairs = Dict{Tuple{Int, Int}, Any}()
            function collect_pair(i, j, pos_diff, distance)
                pairs[(i, j)] = (pos_diff, distance)
            end
            foreach_point_neighbor_(collect_pair, system_coords, neighbor_coords, nhs;
                                    parallelization_backend = SerialBackend())

            return pairs
        end

        # Brute force reference in `Float64`.
        function neighbor_pairs_reference(system_coords, neighbor_coords, search_radius,
                                          periodic_size)
            pairs = Dict{Tuple{Int, Int}, Any}()
            for i in axes(system_coords, 2), j in axes(neighbor_coords, 2)
                pos_diff = system_coords[:, i] - neighbor_coords[:, j]
                if !isnothing(periodic_size)
                    pos_diff -= periodic_size .* round.(pos_diff ./ periodic_size)
                end
                distance = sqrt(sum(abs2, pos_diff))

                if distance <= search_radius
                    pairs[(i, j)] = (pos_diff, distance)
                end
            end

            return pairs
        end

        function test_pairs(pairs, pairs_expected, search_radius)
            @test keys(pairs) == keys(pairs_expected)
            @test all(eltype(pos_diff) == Float32 && distance isa Float32
                      for (pos_diff, distance) in values(pairs))

            # The relative coordinates are rounded to `Float32` once, so the errors are
            # in the order of the machine epsilon relative to the search radius.
            # With `Float32` coordinates, the error would be in the order of
            # `eps(1.0f6) = 0.0625`, which is larger than the search radius.
            atol = 8 * eps(search_radius)
            @test all(isapprox(pairs[key][1], pairs_expected[key][1]; atol) &&
                      isapprox(pairs[key][2], pairs_expected[key][2]; atol)
                      for key in keys(pairs_expected))
        end

        # Points far away from the origin, so that `Float32` coordinates would not be
        # accurate enough.
        offset = 1.0e6
        coords_mixed = [offset .+ 2 * rand(Random.Xoshiro(5), 3, 150),
            offset .+ 2 * rand(Random.Xoshiro(6), 3, 100)]

        @testset "$periodic_name" for periodic_name in ("Non-Periodic", "Periodic")
            if periodic_name == "Periodic"
                # The cell size `2 / 6` is not exactly representable in `Float32`.
                search_radius_ = 0.3f0
                periodic_box = PeriodicBox(min_corner = fill(Float32(offset), 3),
                                           max_corner = fill(Float32(offset + 2), 3))
                periodic_size = 2.0
            else
                search_radius_ = 0.25f0
                periodic_box = nothing
                periodic_size = nothing
            end

            min_corner = fill(offset, 3)
            max_corner = fill(offset + 2, 3)
            template_mixed = SortingGridNeighborhoodSearch{3}(; min_corner, max_corner,
                                                              periodic_box)
            nhs_mixed = CombinedGridNeighborhoodSearch(template_mixed, search_radius_,
                                                       size.(coords_mixed, 2))

            for (point_set, coords_) in enumerate(coords_mixed)
                initialize!(nhs_mixed, point_set, coords_)
            end

            sorted_coords = [coords_[:, point_permutation(grid_nhs_pair(nhs_mixed, i, i))]
                             for (i, coords_) in enumerate(coords_mixed)]

            @testset "Point Set Pair ($i, $j)" for i in 1:2, j in 1:2
                pair = grid_nhs_pair(nhs_mixed, i, j)
                pairs_expected = neighbor_pairs_reference(sorted_coords[i],
                                                          sorted_coords[j],
                                                          search_radius_, periodic_size)

                @testset "$(nameof(loop))" for loop in (foreach_point_neighbor,
                                                foreach_point_neighbor_unsafe)
                    pairs = neighbor_pairs(loop, sorted_coords[i], sorted_coords[j],
                                           pair)
                    test_pairs(pairs, pairs_expected, search_radius_)
                end

                # The coordinates passed to the neighbor loop are not used.
                # Only the relative coordinates stored in the last update are used.
                pairs = neighbor_pairs(foreach_point_neighbor,
                                       fill(NaN, size(sorted_coords[i])),
                                       fill(NaN, size(sorted_coords[j])), pair)
                test_pairs(pairs, pairs_expected, search_radius_)
            end

            @testset "Update" begin
                # Move the (sorted) points of the second point set and sort them again.
                coords_moved = sorted_coords[2] .+
                               0.1 .* randn(Random.Xoshiro(7), size(sorted_coords[2]))
                if periodic_name == "Non-Periodic"
                    coords_moved = clamp.(coords_moved, offset, offset + 2)
                end
                update!(nhs_mixed, 2, coords_moved)
                perm = point_permutation(grid_nhs_pair(nhs_mixed, 2, 2))
                coords_moved = coords_moved[:, perm]

                pair = grid_nhs_pair(nhs_mixed, 1, 2)
                pairs_expected = neighbor_pairs_reference(sorted_coords[1], coords_moved,
                                                          search_radius_, periodic_size)
                pairs = neighbor_pairs(foreach_point_neighbor, sorted_coords[1],
                                       coords_moved, pair)
                test_pairs(pairs, pairs_expected, search_radius_)
            end
        end

        @testset "Same Type" begin
            # With coordinates of the same type as the search radius, the relative
            # coordinates are not used.
            coords32 = Float32.(coords_mixed[1] .- offset)
            template32 = SortingGridNeighborhoodSearch{3}(; min_corner = zeros(Float32, 3),
                                                          max_corner = fill(2.0f0, 3))
            nhs32 = CombinedGridNeighborhoodSearch(template32, 0.25f0, (size(coords32, 2),))
            initialize!(nhs32, 1, coords32)

            @test isempty(nhs32.searches[1].relative_coords)
            @test isempty(nhs32.searches[1].point_cells)

            pair = grid_nhs_pair(nhs32, 1, 1)
            sorted_coords32 = coords32[:, point_permutation(pair)]
            pairs = neighbor_pairs(foreach_point_neighbor, sorted_coords32,
                                   sorted_coords32, pair)
            pairs_expected = neighbor_pairs_reference(sorted_coords32, sorted_coords32,
                                                      0.25f0, nothing)
            @test keys(pairs) == keys(pairs_expected)
        end

        @testset "Less Precise Coordinates" begin
            # With `Float32` coordinates and a `Float64` search radius, the relative
            # coordinates are not used either.
            coords32 = Float32.(coords_mixed[1] .- offset)
            template64 = SortingGridNeighborhoodSearch{3}(; min_corner = zeros(3),
                                                          max_corner = fill(2.0, 3))
            nhs64 = CombinedGridNeighborhoodSearch(template64, 0.25, (size(coords32, 2),))
            initialize!(nhs64, 1, coords32)

            @test isempty(nhs64.searches[1].relative_coords)
            @test isempty(nhs64.searches[1].point_cells)

            pair = grid_nhs_pair(nhs64, 1, 1)
            sorted_coords32 = coords32[:, point_permutation(pair)]
            pairs = neighbor_pairs(foreach_point_neighbor, sorted_coords32,
                                   sorted_coords32, pair)
            pairs_expected = neighbor_pairs_reference(sorted_coords32, sorted_coords32,
                                                      0.25, nothing)
            @test keys(pairs) == keys(pairs_expected)
        end
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
