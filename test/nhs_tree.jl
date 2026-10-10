@testset verbose=true "TreeNeighborhoodSearch" begin
    # Fixed seed to ensure reproducibility of the random point clouds below
    Random.seed!(1)

    # Brute-force reference: all points `j` in `y` with `|x - y_j| <= (h + h_j) / 2`
    function brute_force_neighbors(x, h, y, radii; eachindex_y = axes(y, 2))
        return [j
                for j in eachindex_y if sqrt(sum(abs2, x - y[:, j])) <= (h + radii[j]) / 2]
    end

    # All neighbors of the point `x` with search radius `h` found by the tree NHS, sorted
    function tree_neighbors(nhs, x, h, y)
        neighbors = Int[]

        # The query point is passed as the only column of a coordinate matrix
        foreach_neighbor(reshape(x, :, 1), y, nhs, 1,
                         search_radius = h) do point, neighbor, pos_diff, distance
            push!(neighbors, neighbor)
        end

        return sort(neighbors)
    end

    @testset "Constructor" begin
        cell_list = TreeCellList(min_corner = (0.0, 0.0), max_corner = (1.0, 1.0),
                                 max_level = 3)

        error_str = "a 3D cell list is required for a TreeNeighborhoodSearch{3}"
        @test_throws error_str TreeNeighborhoodSearch{3}(; cell_list)

        error_str = "`TreeNeighborhoodSearch` requires a `TreeCellList`"
        @test_throws error_str TreeNeighborhoodSearch{2}(cell_list = DictionaryCellList{2}())

        error_str = "is not a valid update strategy"
        @test_throws error_str TreeNeighborhoodSearch{2}(; cell_list,
                                                         update_strategy = SemiParallelUpdate())

        error_str = "`max_level` must be between 0 and 20 in 3D"
        @test_throws error_str TreeCellList(min_corner = (0.0, 0.0, 0.0),
                                            max_corner = (1.0, 1.0, 1.0), max_level = 21)

        # The element type is determined by the domain corners
        cell_list = TreeCellList(min_corner = (0.0f0, 0.0f0), max_corner = (1.0f0, 2.0f0),
                                 max_level = 3)
        nhs = TreeNeighborhoodSearch{2}(; cell_list)
        @test eltype(nhs) == Float32

        # A single root cell covering the domain by default
        @test cell_list.root_cell_size == 2.0f0
        @test cell_list.n_roots == (1, 1)

        # A grid of 4 x 8 root cells
        cell_list = TreeCellList(min_corner = (0.0, 0.0), max_corner = (1.0, 2.0),
                                 max_level = 3, root_cell_size = 0.25)
        @test cell_list.n_roots == (4, 8)
    end

    @testset "Morton Order" begin
        # A 2D grid of 3 x 2 root cells with 4 levels below each root cell
        cell_list = TreeCellList(min_corner = (0.0, 0.0), max_corner = (3.0, 2.0),
                                 max_level = 4, root_cell_size = 1.0)

        for level in 0:4
            n_cells = PointNeighbors.n_cells_per_dimension(cell_list, level)
            cells = [Tuple(cell) .- 1 for cell in CartesianIndices(n_cells)]
            indices = [PointNeighbors.cell_index(cell_list, cell, level) for cell in cells]

            # Each cell has a unique zero-based index
            @test sort(vec(indices)) == 0:(prod(n_cells) - 1)

            # The parent of a cell is obtained by removing the last 2 bits of the index
            if level > 0
                for (cell, index) in zip(cells, indices)
                    parent = PointNeighbors.cell_index(cell_list, cell .>> 1, level - 1)
                    @test index >> 2 == parent
                end
            end
        end

        # Within a root cell, the cells are sorted in Z-order
        z_order = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (3, 0), (2, 1), (3, 1)]
        @test [PointNeighbors.cell_index(cell_list, cell, 2) for cell in z_order] == 0:7

        # The second root cell in x-direction starts after the 16 cells of the first one
        @test PointNeighbors.cell_index(cell_list, (4, 0), 2) == 16
    end

    @testset "Tree Structure" begin
        # In 1D, a tree with 3 levels below the root cell [0, 1) has cells of size
        # 1/2, 1/4 and 1/8. We look at the level of the leaf containing each of the
        # 8 finest cells.
        cell_list = TreeCellList(min_corner = (0.0,), max_corner = (1.0,), max_level = 3)
        nhs = TreeNeighborhoodSearch{1}(; cell_list)
        finest_leaf_levels(cell_list) = [PointNeighbors.leaf_level(cell_list, cell)
                                         for cell in 0:7]

        # One point with a small radius on the left and one point with a large radius
        # on the right.
        coords = [0.05 0.95]
        radii = [0.1, 0.2]
        initialize_tree!(nhs, coords, radii)

        # The root cell is split, since its children of size 0.5 are larger than 0.2.
        # Both cells of size 0.5 are split, since all radii are smaller than 0.25.
        # At level 2, the cell [0, 0.25) only contains the point with radius 0.1, so it
        # is split into cells of size 0.125. The cell [0.75, 1) contains the point
        # with radius 0.2 and can't be split. The other cells are empty.
        @test finest_leaf_levels(cell_list) == [3, 3, 2, 2, 2, 2, 2, 2]

        # Now increase the radius of the right point to 0.4.
        # The cell [0, 0.5) is not split anymore, although it only contains a point with
        # a small radius. The large radius in the neighboring cell [0.5, 1) could
        # otherwise reach beyond the 3^d block around the leaf containing the left point.
        radii = [0.1, 0.4]
        update_tree!(nhs, coords, radii)
        @test finest_leaf_levels(cell_list) == [1, 1, 1, 1, 1, 1, 1, 1]

        # With a capacity of 1 point per cell, only the root cell with 2 points is split
        cell_list = TreeCellList(min_corner = (0.0,), max_corner = (1.0,), max_level = 3,
                                 capacity = 1)
        nhs = TreeNeighborhoodSearch{1}(; cell_list)
        initialize_tree!(nhs, coords, [0.1, 0.2])
        @test finest_leaf_levels(cell_list) == [1, 1, 1, 1, 1, 1, 1, 1]
    end

    @testset "Points Sorted Into Leaves" begin
        # Random points in 2D with radii growing in x-direction
        coords = rand(2, 500)
        radii = 0.01 .+ 0.1 .* coords[1, :]

        cell_list = TreeCellList(min_corner = (0.0, 0.0), max_corner = (1.0, 1.0),
                                 max_level = 6)
        nhs = TreeNeighborhoodSearch{2}(; cell_list)

        # Leave out some points
        eachindex_y = 1:2:500
        initialize_tree!(nhs, coords, radii; eachindex_y)

        # Loop over all leaves by jumping from leaf to leaf over the finest cells
        n_finest_cells = 4^6
        cell = 0
        points_found = Int[]
        while cell < n_finest_cells
            level = PointNeighbors.leaf_level(cell_list, cell)
            next_cell = cell + PointNeighbors.n_finest_cells(cell_list, level)

            points = PointNeighbors.points_in_finest_cells(cell_list, cell, next_cell)
            for point in points
                # The point must be inside this leaf
                cell_size = 1 / 2^level
                leaf_cell = floor.(Int, coords[:, point] ./ cell_size)
                @test PointNeighbors.cell_index(cell_list, Tuple(leaf_cell), level) ==
                      cell >> (2 * (6 - level))
            end
            append!(points_found, points)

            cell = next_cell
        end

        # Each active point is stored in exactly one leaf
        @test sort(points_found) == eachindex_y
    end

    @testset verbose=true "Compare Against Brute Force $(NDIMS)D" for NDIMS in 1:3
        n_points = (200, 2_000, 4_000)[NDIMS]
        max_level = (8, 7, 5)[NDIMS]
        coords = rand(NDIMS, n_points)

        # Move the first point to the center of the unit cube, which is on a cell boundary
        # on every level.
        center = fill(0.5, NDIMS)
        coords[:, 1] .= center

        # Different distributions of search radii in the unit cube
        radius_distributions = [
            "Smooth" => x -> 0.01 + 0.1 * x[1]^2,
            # A jump by a factor of 10 at a cell boundary, which violates any 2:1 balance
            # of the tree. Fine leaves on one side are next to coarse leaves on the other.
            "Jump" => x -> x[NDIMS] < 0.5 ? 0.02 : 0.2,
            # A single point with a large radius in the center surrounded by small radii.
            # Only the leaves around this point must be coarse.
            # The first point is moved to the center below.
            "Isolated Large Radius" => x -> x == center ? 0.25 : 0.02,
            # Arbitrary radii without any spatial correlation
            "Random" => x -> 0.01 + 0.2 * rand(),
            "Uniform" => x -> 0.05
        ]

        # Different tree configurations
        cell_lists = [
            "Single Root Cell" => TreeCellList(; min_corner = zeros(NDIMS),
                                               max_corner = ones(NDIMS), max_level),
            "Capacity 8" => TreeCellList(; min_corner = zeros(NDIMS),
                                         max_corner = ones(NDIMS), max_level, capacity = 8),
            # Multiple root cells, which must be larger than all radii
            "Root Grid" => TreeCellList(; min_corner = zeros(NDIMS),
                                        max_corner = ones(NDIMS), max_level = max_level - 2,
                                        root_cell_size = 0.25),
            # A domain that is larger than the point cloud
            "Large Domain" => TreeCellList(; min_corner = fill(-1.0, NDIMS),
                                           max_corner = fill(2.0, NDIMS), max_level)
        ]

        @testset "$cell_list_name, $radii_name Radii" for (cell_list_name, cell_list) in
                                                          cell_lists,
                                                          (radii_name, radius) in
                                                          radius_distributions

            radii = [radius(coords[:, j]) for j in axes(coords, 2)]
            nhs = TreeNeighborhoodSearch{NDIMS}(; cell_list)

            # Initialize with different coordinates first to also test the update
            initialize_tree!(nhs, rand(NDIMS, n_points), radii)
            update_tree!(nhs, coords, radii)

            # Query each point of the tree with its own radius
            @test all(axes(coords, 2)) do i
                tree_neighbors(nhs, coords[:, i], radii[i], coords) ==
                brute_force_neighbors(coords[:, i], radii[i], coords, radii)
            end

            # Query arbitrary points with arbitrary radii, including points outside of
            # the domain and radii larger than the domain.
            # Only larger domains are supported with multiple root cells.
            max_query_radius = cell_list_name == "Root Grid" ? 0.25 : 2.0
            query_coords = 1.6 .* rand(NDIMS, 200) .- 0.3
            query_radii = max_query_radius .* rand(200) .^ 2
            @test all(axes(query_coords, 2)) do i
                x = query_coords[:, i]
                tree_neighbors(nhs, x, query_radii[i], coords) ==
                brute_force_neighbors(x, query_radii[i], coords, radii)
            end
        end
    end

    @testset "Neighbor Loops" begin
        # Points in 2D with a smooth radius distribution
        coords = rand(2, 1000)
        radii = 0.02 .+ 0.1 .* coords[1, :] .* coords[2, :]

        cell_list = TreeCellList(min_corner = (0.0, 0.0), max_corner = (1.0, 1.0),
                                 max_level = 7)
        nhs = TreeNeighborhoodSearch{2}(; cell_list)
        initialize_tree!(nhs, coords, radii)

        expected = [brute_force_neighbors(coords[:, i], radii[i], coords, radii)
                    for i in axes(coords, 2)]

        # `foreach_neighbor_unsafe` with the correct `pos_diff` and `distance`
        neighbors = [Int[] for _ in axes(coords, 2)]
        for point in axes(coords, 2)
            foreach_neighbor_unsafe(coords, coords, nhs, point,
                                    search_radius = radii[point]) do point, neighbor,
                                                                     pos_diff, distance
                @test pos_diff ≈ coords[:, point] - coords[:, neighbor]
                @test distance ≈ sqrt(sum(abs2, pos_diff))
                push!(neighbors[point], neighbor)
            end
        end
        @test sort.(neighbors) == expected

        # `mapreduce_neighbor` and `mapreduce_neighbor_unsafe` summing up neighbor indices
        neighbor_sums = map(axes(coords, 2)) do point
            mapreduce_neighbor(+, coords, coords, nhs, point; init = 0,
                               search_radius = radii[point]) do point, neighbor, _, _
                neighbor
            end
        end
        @test neighbor_sums == sum.(expected)

        neighbor_sums = map(axes(coords, 2)) do point
            mapreduce_neighbor_unsafe(+, coords, coords, nhs, point; init = 0,
                                      search_radius = radii[point]) do point, neighbor, _, _
                neighbor
            end
        end
        @test neighbor_sums == sum.(expected)

        # The query must not allocate
        function allocations_count_neighbors(coords, nhs, point, search_radius)
            @allocated(mapreduce_neighbor((point, neighbor, pos_diff, distance) -> 1,
                                          +, coords, coords, nhs, point;
                                          init = 0, search_radius))
        end
        allocations_count_neighbors(coords, nhs, 1, radii[1])
        @test allocations_count_neighbors(coords, nhs, 1, radii[1]) == 0

        # Without points in the tree, `init` is returned unchanged
        empty_coords = zeros(2, 0)
        initialize_tree!(nhs, empty_coords, Float64[])
        result = mapreduce_neighbor((_...) -> error("no neighbors expected"), +,
                                    coords, empty_coords, nhs, 1; init = 123,
                                    search_radius = 0.1)
        @test result == 123
    end

    @testset "Inactive Points" begin
        coords = rand(3, 1000)
        radii = 0.05 .+ 0.1 .* coords[3, :]

        cell_list = TreeCellList(min_corner = (0.0, 0.0, 0.0),
                                 max_corner = (1.0, 1.0, 1.0), max_level = 5)
        nhs = TreeNeighborhoodSearch{3}(; cell_list)

        # Only every third point is part of the tree
        eachindex_y = 1:3:1000
        initialize_tree!(nhs, coords, radii; eachindex_y)

        @test all(axes(coords, 2)) do i
            tree_neighbors(nhs, coords[:, i], radii[i], coords) ==
            brute_force_neighbors(coords[:, i], radii[i], coords, radii; eachindex_y)
        end
    end

    @testset "`copy_neighborhood_search`" begin
        cell_list = TreeCellList(min_corner = (0.0, 0.0), max_corner = (1.0, 2.0),
                                 max_level = 4, root_cell_size = 0.5, capacity = 3)
        nhs = TreeNeighborhoodSearch{2}(; cell_list, update_strategy = SerialUpdate())

        # The search radius is ignored, as the radii are passed to `initialize_tree!`
        copy = copy_neighborhood_search(nhs, 1.0, 10)

        @test copy.cell_list.min_corner == cell_list.min_corner
        @test copy.cell_list.max_corner == cell_list.max_corner
        @test copy.cell_list.max_level == 4
        @test copy.cell_list.root_cell_size == 0.5
        @test copy.cell_list.capacity == 3
        @test copy.update_strategy == SerialUpdate()

        # The copy has its own data structures
        @test copy.cell_list.leaf_level !== cell_list.leaf_level
    end

    @testset "Errors" begin
        cell_list = TreeCellList(min_corner = (0.0, 0.0), max_corner = (1.0, 1.0),
                                 max_level = 3)
        nhs = TreeNeighborhoodSearch{2}(; cell_list)
        coords = [0.1 0.5; 0.2 0.5]

        # The radii have to be passed with `initialize_tree!`
        @test_throws "Use `initialize_tree!` instead" initialize!(nhs, coords, coords)
        @test_throws "Use `update_tree!` instead" update!(nhs, coords, coords)

        error_str = "the number of search radii must match the number of points"
        @test_throws error_str initialize_tree!(nhs, coords, [0.1])

        error_str = "particle coordinates are NaN or outside the domain bounds"
        @test_throws error_str initialize_tree!(nhs, [0.1 1.5; 0.2 0.5], [0.1, 0.1])
        @test_throws error_str initialize_tree!(nhs, [0.1 NaN; 0.2 0.5], [0.1, 0.1])

        # The search radius of each query point is required.
        # Therefore, `foreach_point_neighbor` is not supported.
        initialize_tree!(nhs, coords, [0.1, 0.1])
        error_str = "requires the search radius of the query point"
        @test_throws error_str foreach_neighbor((_...) -> nothing, coords, coords, nhs, 1)
        @test_throws "`foreach_point_neighbor` is not supported" foreach_point_neighbor((_...) -> nothing,
                                                                                        coords,
                                                                                        coords,
                                                                                        nhs)

        # With multiple root cells, the radii must not exceed the root cell size
        cell_list = TreeCellList(min_corner = (0.0, 0.0), max_corner = (1.0, 1.0),
                                 max_level = 3, root_cell_size = 0.5)
        nhs = TreeNeighborhoodSearch{2}(; cell_list)

        error_str = "the search radii must not exceed the `root_cell_size`"
        @test_throws error_str initialize_tree!(nhs, coords, [0.1, 0.6])
    end
end;
