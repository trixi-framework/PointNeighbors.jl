@testset verbose=true "TreeNeighborhoodSearch" begin
    # Helper function to bypass the need for an iterator in tests
    function collect_tree_neighbors(point_coords, nhs, coords_matrix)
        neighbors = Int[]

        # Define the closure f(point, neighbor, pos_diff, distance)
        f = (point, neighbor, pos_diff, dist) -> push!(neighbors, neighbor)
        search_radius_with_tol = nhs.search_radius + 1e-10

        # We pass 1 as a dummy 'point' index since we are querying by raw coordinates
        PointNeighbors.foreach_neighbor_inner(f, coords_matrix, nhs, 1, point_coords,
                                              search_radius_with_tol)

        return sort(unique(neighbors))
    end

    @testset "`initialize_tree!` with dynamic merging" begin
        # 2D coordinates designed to fall into specific Morton indices for an 8x8 grid (max_level = 3)
        # Z=1 : (0.0, 0.0) Point 1
        # Z=2 : (1.0, 0.0) Point 2
        # Z=3 : (0.0, 1.0) Point 3
        # Z=33: (0.0, 5.0) Point 4
        # Z=34: (1.0, 5.0) Point 5
        # Z=64: (8.0, 8.0) Point 6
        coordindates = [0.0 1.0 0.0 0.0 1.0 8.0;
                        0.0 0.0 1.0 5.0 5.0 8.0]

        min_corner = (0.0, 0.0)
        max_corner = (8.0, 8.0)
        n_dims, n_points = size(coordindates)

        # capacity_per_cell = 2 ensures some cells merge while denser areas don't
        cell_list = TreeGridCellList(; min_corner, max_corner, max_level = 3,
                                     capacity_per_cell = 2)

        (; cell_levels, cells, max_level) = cell_list
        length_grid = PointNeighbors.grid_length(cell_list)
        cell_length = length_grid / (2^max_level)
        search_radius = 2 * cell_length

        nhs = TreeNeighborhoodSearch{n_dims}(; cell_list, search_radius, n_points)

        # Merge up to the highest hierarchical level (level 0)
        PointNeighbors.initialize_tree!(nhs, coordindates, iters = max_level)

        # Assertions for unmerged cells at max_level = 3
        # The coarse level 1 block (indices 1-16) has 3 points total, so its sub-blocks cannot merge fully.
        @test cells[1] == [1]
        @test cells[2] == [2]
        @test cells[3] == [3]
        @test cell_levels[1] == 3
        @test cell_levels[2] == 3
        @test cell_levels[3] == 3
        @test cell_levels[4] == -1  # Cell 4 received no points and wasn't merged into

        # Assertions for intermediate merged empty cells at level 2
        # Blocks 5-8, 9-12, 13-16 have 0 points, so they successfully merge into their level 2 bases
        @test cell_levels[5] == 2
        @test cell_levels[9] == 2
        @test cell_levels[13] == 2
        @test isempty(cells[5])

        # Assertions for merged cells at level 1
        # Block 17-32 is completely empty, merging up to its base index 17 at level 1
        @test cell_levels[17] == 1
        @test isempty(cells[17])

        # Block 33-48 has exactly 2 points, merging fully to its base index 33 at level 1
        @test sort(cells[33]) == [4, 5]
        @test cell_levels[33] == 1
        @test cell_levels[34] == -1 # Subcell is successfully reset upon merging

        # Block 49-64 has only 1 point, merging all the way down to base index 49 at level 1
        @test cells[49] == [6]
        @test cell_levels[49] == 1
        @test cell_levels[64] == -1 # Original fine cell is successfully reset
    end

    @testset "`merge`" begin
        # 2D coordinates mapping to Morton Z-indices: 1, 2, 3, 33, 34, 64
        coords = [0.0 1.0 0.0 0.0 1.0 8.0;
                  0.0 0.0 1.0 5.0 5.0 8.0]
        min_corner = (0.0, 0.0)
        max_corner = (8.0, 8.0)
        n_dims, n_points = size(coords)

        cell_list = TreeGridCellList(; min_corner, max_corner, max_level = 3,
                                     capacity_per_cell = 2)

        (; cell_levels, cells, max_level) = cell_list
        length_grid = PointNeighbors.grid_length(cell_list)
        cell_length = length_grid / (2^max_level)
        search_radius = 2 * cell_length

        nhs = TreeNeighborhoodSearch{n_dims}(; cell_list, search_radius, n_points)

        # Place points at the finest level without automatically merging
        PointNeighbors.initialize_tree!(nhs, coords, iters = 0)

        # Assert initial state
        @test cells[1] == [1]
        @test cells[2] == [2]
        @test cells[3] == [3]
        @test cells[33] == [4]
        @test cells[34] == [5]
        @test cells[64] == [6]
        @test all(cell_levels[[1, 2, 3, 33, 34, 64]] .== 3)

        # Merge at Level 2
        # Offsets for level 2 are 4 (e.g., base indices 1, 5, 9, ..., 33, ..., 61)
        PointNeighbors.mark_merge!(nhs, level = 2, capacity = 2)
        PointNeighbors.apply_merge!(nhs, level = 2)

        @test cells[1] == [1]           # Did not merge
        @test cells[2] == [2]
        @test cells[3] == [3]
        @test sort(cells[33]) == [4, 5] # Merged into base 33
        @test cells[61] == [6]          # Merged into base 61

        @test all(cell_levels[[1, 2, 3]] .== 3)
        @test cell_levels[33] == 2
        @test cell_levels[34] == -1     # Subcell is successfully reset
        @test cell_levels[61] == 2
        @test cell_levels[64] == -1     # Subcell is successfully reset

        # Merge at Level 1
        # Offsets for level 1 are 16 (e.g., base indices 1, 17, 33, 49)
        PointNeighbors.mark_merge!(nhs, level = 1, capacity = 2)
        PointNeighbors.apply_merge!(nhs, level = 1)

        @test cells[1] == [1]           # Still did not merge (Base 1 has 3 total points)
        @test sort(cells[33]) == [4, 5] # Merged again, this time to level 1 base 33!
        @test cells[49] == [6]          # Merged into level 1 base 49!

        @test all(cell_levels[[1, 2, 3]] .== 3)
        @test cell_levels[33] == 1
        @test cell_levels[49] == 1
        @test cell_levels[61] == -1     # Reset from previous level 2
    end

    @testset "`refine`" begin
        # 2D coordinates mapping to Morton Z-indices: 1, 2, 3, 33, 34, 64
        coords = [0.0 1.0 0.0 0.0 1.0 8.0;
                  0.0 0.0 1.0 5.0 5.0 8.0]
        min_corner = (0.0, 0.0)
        max_corner = (8.0, 8.0)
        n_dims, n_points = size(coords)

        cell_list = TreeGridCellList(; min_corner, max_corner, max_level = 3,
                                     capacity_per_cell = 2)

        (; cell_levels, cells, max_level) = cell_list
        length_grid = PointNeighbors.grid_length(cell_list)
        cell_length = length_grid / (2^max_level)
        search_radius = 2 * cell_length

        nhs = TreeNeighborhoodSearch{n_dims}(; cell_list, search_radius, n_points)

        # Setup root cell manually (Level 0)
        PointNeighbors.empty!(cell_list)
        cell_levels .= -1
        cell_levels[1] = 0
        for i in 1:6
            PointNeighbors.push_cell!(cell_list, 1, i)
        end

        # Refine Level 0, root splits into Level 1 cells at offsets 1, 17, 33, 49
        PointNeighbors.mark_refine!(nhs, level = 0, capacity = 2)
        PointNeighbors.apply_refine!(nhs, coords, level = 0)

        @test sort(cells[1]) == [1, 2, 3] # P1, P2, P3 fall into subcell 1
        @test sort(cells[33]) == [4, 5]   # P4, P5 fall into subcell 33
        @test cells[49] == [6]            # P6 falls into subcell 49

        @test cell_levels[1] == 1
        @test cell_levels[33] == 1
        @test cell_levels[49] == 1

        # Refine Level 1
        # Cell 1 exceeds capacity (3 > 2), so it splits into Level 2 cells at offsets 1, 5, 9, 13
        PointNeighbors.mark_refine!(nhs, level = 1, capacity = 2)
        PointNeighbors.apply_refine!(nhs, coords, level = 1)

        @test sort(cells[1]) == [1, 2, 3] # They still fall into the first spatial quadrant of cell 1
        @test cell_levels[1] == 2

        # Cells 33 and 49 do not refine because their point count <= capacity
        @test sort(cells[33]) == [4, 5]
        @test cell_levels[33] == 1
        @test cells[49] == [6]
        @test cell_levels[49] == 1

        # Refine Level 2
        # Cell 1 still exceeds capacity (3 > 2), splits into Level 3 cells at offsets 1, 2, 3, 4
        PointNeighbors.mark_refine!(nhs, level = 2, capacity = 2)
        PointNeighbors.apply_refine!(nhs, coords, level = 2)

        # At the finest level, they fall into individual Morton indices
        @test cells[1] == [1]
        @test cells[2] == [2]
        @test cells[3] == [3]

        @test cell_levels[1] == 3
        @test cell_levels[2] == 3
        @test cell_levels[3] == 3

        # Verify that previously unrefined coarse cells remain unchanged
        @test sort(cells[33]) == [4, 5]
        @test cell_levels[33] == 1
        @test cells[49] == [6]
        @test cell_levels[49] == 1
    end

    @testset "`foreach_neighbor`" begin
        # 2D coordinates mapping to Morton Z-indices: 1, 2, 3, 33, 34, 64
        coords = [0.0 1.0 0.0 0.0 1.0 8.0;
                  0.0 0.0 1.0 5.0 5.0 8.0]
        min_corner = (0.0, 0.0)
        max_corner = (8.0, 8.0)
        n_dims, n_points = size(coords)

        cell_list = TreeGridCellList(; min_corner, max_corner, max_level = 3,
                                     capacity_per_cell = 1)

        (; cell_levels, cells, max_level) = cell_list
        length_grid = PointNeighbors.grid_length(cell_list)
        cell_length = length_grid / (2^max_level)
        search_radius = 2 * cell_length # Radius = 2.0

        nhs = TreeNeighborhoodSearch{n_dims}(; cell_list, search_radius, n_points)
        PointNeighbors.initialize_tree!(nhs, coords, iters = 0)

        neighbors = Vector{Int}(undef, 0)
        point = 1
        point_coords = coords[:, point]
        PointNeighbors.foreach_neighbor(coords, nhs, point, point_coords,
                                        nhs.search_radius) do particle, neighbor, pos_diff,
                                                              distance
            push!(neighbors, neighbor)
        end

        # P1(0,0), P2(1,0), P3(0,1) are all within radius 2.0. P4(0,5) is not.
        @test sort!(neighbors) == [1, 2, 3]
    end
    @testset "`update!`" begin end

    @testset "`foreach_neighbor` nested" begin
        # Generic helper to collect and sort neighbors from any NHS implementation
        function collect_neighbors(coords, nhs, point)
            neighbors = Vector{Int}()
            PointNeighbors.foreach_neighbor(coords, nhs, point, coords[:, point],
                                            nhs.search_radius) do particle, neighbor,
                                                                  pos_diff, distance
                push!(neighbors, neighbor)
            end
            return sort!(neighbors)
        end

        @testset "2D" begin
            coords = [0.0 1.0 0.0 0.0 1.0 8.0;
                      0.0 0.0 1.0 5.0 5.0 8.0]
            min_corner = (0.0, 0.0)
            max_corner = (8.0, 8.0)
            n_dims, n_points = size(coords)

            cell_list = TreeGridCellList(; min_corner, max_corner, max_level = 3,
                                         capacity_per_cell = 1)

            length_grid = PointNeighbors.grid_length(cell_list)
            cell_length = length_grid / (2^cell_list.max_level)
            search_radius = 2 * cell_length

            nhs_tree = TreeNeighborhoodSearch{n_dims}(; cell_list, search_radius, n_points)
            nhs_trivial = TrivialNeighborhoodSearch{n_dims}(; search_radius,
                                                            eachpoint = 1:n_points)

            PointNeighbors.initialize_tree!(nhs_tree, coords, iters = 0)

            # Verify all points against TrivialNeighborhoodSearch
            for point in 1:n_points
                @test collect_neighbors(coords, nhs_tree, point) ==
                      collect_neighbors(coords, nhs_trivial, point)
            end

            # Explicit Boundary Check: Point 6 is at the top-right corner (8.0, 8.0)
            @test collect_neighbors(coords, nhs_tree, 6) ==
                  collect_neighbors(coords, nhs_trivial, 6)
        end

        @testset "2D with Merge (`iters > 0`)" begin
            coords = [0.0 1.0 0.0 0.0 1.0 8.0;
                      0.0 0.0 1.0 5.0 5.0 8.0]
            min_corner = (0.0, 0.0)
            max_corner = (8.0, 8.0)
            n_dims, n_points = size(coords)

            # Capacity 2 ensures Points 4 and 5 merge into a coarse block, testing the find_coarse_ancestor logic during neighbor search
            cell_list = TreeGridCellList(; min_corner, max_corner, max_level = 3,
                                         capacity_per_cell = 2)

            length_grid = PointNeighbors.grid_length(cell_list)
            cell_length = length_grid / (2^cell_list.max_level)
            search_radius = 2 * cell_length

            nhs_tree = TreeNeighborhoodSearch{n_dims}(; cell_list, search_radius, n_points)
            nhs_trivial = TrivialNeighborhoodSearch{n_dims}(; search_radius,
                                                            eachpoint = 1:n_points)

            # Perform 2 merge iterations to collapse empty or sparse cells
            PointNeighbors.initialize_tree!(nhs_tree, coords, iters = 2)

            for point in 1:n_points
                @test collect_neighbors(coords, nhs_tree, point) ==
                      collect_neighbors(coords, nhs_trivial, point)
            end
        end

        @testset "2D Multiple Search Radii" begin
            coords = [0.0 1.0 0.0 0.0 1.0 8.0;
                      0.0 0.0 1.0 5.0 5.0 8.0]
            min_corner = (0.0, 0.0)
            max_corner = (8.0, 8.0)
            n_dims, n_points = size(coords)

            cell_list = TreeGridCellList(; min_corner, max_corner, max_level = 3,
                                         capacity_per_cell = 1)
            length_grid = PointNeighbors.grid_length(cell_list)
            cell_length = length_grid / (2^cell_list.max_level)

            # Test small search radius (0.5 * 1.0 = 0.5)
            small_radius = 0.5 * cell_length
            nhs_small_tree = TreeNeighborhoodSearch{n_dims}(; cell_list,
                                                            search_radius = small_radius,
                                                            n_points)
            nhs_small_trivial = TrivialNeighborhoodSearch{n_dims}(;
                                                                  search_radius = small_radius,
                                                                  eachpoint = 1:n_points)
            PointNeighbors.initialize_tree!(nhs_small_tree, coords, iters = 0)

            @test collect_neighbors(coords, nhs_small_tree, 1) ==
                  collect_neighbors(coords, nhs_small_trivial, 1)

            # Test large search radius (4.0 * 1.0 = 4.0)
            large_radius = 4.0 * cell_length
            nhs_large_tree = TreeNeighborhoodSearch{n_dims}(; cell_list,
                                                            search_radius = large_radius,
                                                            n_points)
            nhs_large_trivial = TrivialNeighborhoodSearch{n_dims}(;
                                                                  search_radius = large_radius,
                                                                  eachpoint = 1:n_points)
            PointNeighbors.initialize_tree!(nhs_large_tree, coords, iters = 0)

            for point in 1:n_points
                @test collect_neighbors(coords, nhs_large_tree, point) ==
                      collect_neighbors(coords, nhs_large_trivial, point)
            end
        end

        @testset "3D" begin
            # Using an equivalent predictable Z-order layout in 3D
            coords_3d = [0.0 1.0 0.0 0.0 8.0;
                         0.0 0.0 1.0 5.0 8.0;
                         0.0 0.0 0.0 1.0 8.0]
            min_corner = (0.0, 0.0, 0.0)
            max_corner = (8.0, 8.0, 8.0)
            n_dims, n_points = size(coords_3d)

            cell_list = TreeGridCellList(; min_corner, max_corner, max_level = 3,
                                         capacity_per_cell = 1)
            length_grid = PointNeighbors.grid_length(cell_list)
            cell_length = length_grid / (2^cell_list.max_level)
            search_radius = 2.5 * cell_length

            nhs_3d_tree = TreeNeighborhoodSearch{n_dims}(; cell_list, search_radius,
                                                         n_points)
            nhs_3d_trivial = TrivialNeighborhoodSearch{n_dims}(; search_radius,
                                                               eachpoint = 1:n_points)
            PointNeighbors.initialize_tree!(nhs_3d_tree, coords_3d, iters = 0)

            for point in 1:n_points
                @test collect_neighbors(coords_3d, nhs_3d_tree, point) ==
                      collect_neighbors(coords_3d, nhs_3d_trivial, point)
            end
        end
    end

    @testset "Constructor" begin
        error_str = "a 2D cell list is required for a TreeNeighborhoodSearch{2}"
        cell_list_3d = TreeGridCellList(min_corner = (0.0, 0.0, 0.0),
                                        max_corner = (1.0, 1.0, 1.0))
        @test_throws error_str TreeNeighborhoodSearch{2}(cell_list = cell_list_3d)

        error_str = "is not a valid update strategy"
        cell_list_2d = TreeGridCellList(min_corner = (0.0, 0.0), max_corner = (1.0, 1.0))
        @test_throws "test $error_str" TreeNeighborhoodSearch{2}(cell_list = cell_list_2d,
                                                                 update_strategy = :test)

        error_str = "Current implementation of `TreeNeighborhoodSearch` only supports `TreeGridCellList`"
        @test_throws error_str TreeNeighborhoodSearch{2}(cell_list = DictionaryCellList{2}())

        nhs = TreeNeighborhoodSearch{2}(cell_list = cell_list_2d,
                                        update_strategy = ParallelUpdate())
        nhs2 = @trixi_test_nowarn PointNeighbors.Adapt.adapt_structure(Array, nhs)

        @test nhs2.update_strategy == nhs.update_strategy
    end

    @testset "`copy_neighborhood_search" begin
        # Basic copy
        min_corner = (0.0, 0.0)
        max_corner = (1.0, 1.0)
        cell_list = TreeGridCellList(; min_corner, max_corner, max_level = 3)

        nhs = TreeNeighborhoodSearch{2}(cell_list = cell_list,
                                        update_strategy = ParallelUpdate())
        copy = copy_neighborhood_search(nhs, 1.0, 10)

        @test ndims(copy) == 2
        @test PointNeighbors.search_radius(copy) == 1.0
        @test copy.cell_list isa TreeGridCellList
        @test copy.update_strategy == ParallelUpdate()
        @test copy.cell_list.max_level == 3
    end

    @testset "Rectangular Point Cloud 2D" begin
        #### Setup
        # Rectangle of equidistantly spaced points
        # from (x, y) = (-0.25, -0.25) to (x, y) = (0.35, 0.35).
        range = -0.25:0.1:0.35
        coordinates1 = hcat(collect.(Iterators.product(range, range))...)
        n_points = size(coordinates1, 2)

        point_position1 = [0.05, 0.05]
        search_radius = 0.1

        # The TreeGridCellList needs a pre-defined domain that encompasses
        # the points' starting locations and their subsequent movements.
        min_corner = (-1.0, -4.0)
        max_corner = (2.0, 1.0)
        cell_list = TreeGridCellList(; min_corner, max_corner, max_level = 5,
                                     capacity_per_cell = 4)

        # Create neighborhood search
        nhs1 = TreeNeighborhoodSearch{2}(; search_radius, n_points, cell_list)

        initialize!(nhs1, coordinates1, coordinates1)

        # Get each neighbor for `point_position1`
        neighbors1 = collect_tree_neighbors(point_position1, nhs1, coordinates1)

        # Move points
        coordinates2 = coordinates1 .+ [1.4, -3.5]

        # Update neighborhood search
        update!(nhs1, coordinates2, coordinates2)

        # Get each neighbor for updated NHS
        neighbors2 = collect_tree_neighbors(point_position1, nhs1, coordinates2)

        # Change position
        point_position2 = point_position1 .+ [1.4, -3.5]

        # Get each neighbor for `point_position2`
        neighbors3 = collect_tree_neighbors(point_position2, nhs1, coordinates2)

        # Double search radius
        cell_list_double = TreeGridCellList(; min_corner, max_corner, max_level = 5)
        nhs2 = TreeNeighborhoodSearch{2}(search_radius = 2 * search_radius,
                                         n_points = size(coordinates1, 2),
                                         cell_list = cell_list_double)
        initialize!(nhs2, coordinates1, coordinates1)

        # Get each neighbor in double search radius
        neighbors4 = collect_tree_neighbors(point_position1, nhs2, coordinates1)

        # Move points
        coordinates2 = coordinates1 .+ [0.4, -0.4]

        # Update neighborhood search
        update!(nhs2, coordinates2, coordinates2)

        # Get each neighbor in double search radius
        neighbors5 = collect_tree_neighbors(point_position1, nhs2, coordinates2)

        # Only the orthogonal neighbors (up, down, left, right) + self are <= 0.1 distance
        @test neighbors1 == [18, 24, 25, 26, 32]

        @test neighbors2 == Int[]

        @test neighbors3 == [18, 24, 25, 26, 32]

        # Points strictly inside a radius of 0.2
        @test neighbors4 == [11, 17, 18, 19, 23, 24, 25, 26, 27, 31, 32, 33, 39]

        # Only point 43 shifted by (0.4, -0.4) ends up at distance <= 0.2 from (0.05, 0.05)
        @test neighbors5 == [43]
    end

    @testset verbose=true "Rectangular Point Cloud 3D" begin
        #### Setup
        # Rectangle of equidistantly spaced points
        # from (x, y, z) = (-0.25, -0.25, -0.25) to (x, y, z) = (0.35, 0.35, 0.35).
        range = -0.25:0.1:0.35
        coordinates1 = hcat(collect.(Iterators.product(range, range, range))...)
        n_points = size(coordinates1, 2)

        point_position1 = [0.05, 0.05, 0.05]
        search_radius = 0.1

        # Bounding box containing both initial coordinates and offset translations
        min_corner = (-1.0, -4.0, -1.0)
        max_corner = (2.0, 1.0, 2.0)
        cell_list = TreeGridCellList(; min_corner, max_corner, max_level = 5,
                                     capacity_per_cell = 4)

        # Create neighborhood search
        nhs1 = TreeNeighborhoodSearch{3}(; search_radius, n_points, cell_list)
        initialize!(nhs1, coordinates1, coordinates1)

        # Get each neighbor for `point_position1`
        neighbors1 = collect_tree_neighbors(point_position1, nhs1, coordinates1)

        # Move points
        coordinates2 = coordinates1 .+ [1.4, -3.5, 0.8]

        # Update neighborhood search
        update!(nhs1, coordinates2, coordinates2)

        # Get each neighbor for updated NHS
        neighbors2 = collect_tree_neighbors(point_position1, nhs1, coordinates2)

        # Change position
        point_position2 = point_position1 .+ [1.4, -3.5, 0.8]

        # Get each neighbor for `point_position2`
        neighbors3 = collect_tree_neighbors(point_position2, nhs1, coordinates2)

        # 6 orthogonal neighbors + self in 3D
        @test neighbors1 == [123, 165, 171, 172, 173, 179, 221]

        @test neighbors2 == Int[]

        @test neighbors3 == [123, 165, 171, 172, 173, 179, 221]

        update_strategies = (ParallelUpdate(),)
        @testset verbose=true "eachindex_y $update_strategy" for update_strategy in
                                                                 update_strategies
            # Test that `eachindex_y` is passed correctly to the neighborhood search.
            # Tree NHS requires calculating min and max bounds for the shared container.
            # `vec` ensures the 3x1 Matrix safely converts to a Tuple.
            min_c = Tuple(vec(min.(minimum(coordinates1, dims = 2),
                                   minimum(coordinates2, dims = 2))))
            max_c = Tuple(vec(max.(maximum(coordinates1, dims = 2),
                                   maximum(coordinates2, dims = 2))))

            cell_list_subset = TreeGridCellList(; min_corner = min_c, max_corner = max_c,
                                                max_level = 5, capacity_per_cell = 4)

            nhs2 = TreeNeighborhoodSearch{3}(; search_radius, n_points, update_strategy,
                                             cell_list = cell_list_subset)

            # Initialize with all points
            initialize!(nhs2, coordinates1, coordinates1)

            # Update with a subset of points
            update!(nhs2, coordinates2, coordinates2; eachindex_y = 120:220)

            neighbors2 = collect_tree_neighbors(point_position1, nhs2, coordinates2)
            neighbors3 = collect_tree_neighbors(point_position2, nhs2, coordinates2)

            # Check that the neighbors are the intersection of the previous neighbors
            # with the `eachindex_y` range. (Note: 221 is excluded because 221 > 220)
            @test neighbors2 == Int[]
            @test neighbors3 == [123, 165, 171, 172, 173, 179]
        end
    end

    function plot_active_tree(cell_list, coords; zoom_to_points::Bool = false,
                              zoom_padding = 0.2)
        (; cell_levels, min_corner, max_level) = cell_list

        # Calculate base dimensions using the internal utilities
        L_grid = PointNeighbors.grid_length(cell_list) #[cite: 1]
        finest_cell_length = L_grid / (2^max_level)

        # Initialize the plot
        p = plot(aspect_ratio = :equal, legend = :outertopright,
                 title = "TreeGridCellList Active Cells",
                 framestyle = :box)

        # Plot each active cell
        for i in eachindex(cell_levels)
            L = cell_levels[i]

            # If the level is != -1, the cell is active (either merged or a finest leaf)
            if L != -1 #[cite: 2]
                cell_size = L_grid / (2^L)

                # Get the 2D Cartesian index (1-based) of the base finest-cell 
                cartesian = PointNeighbors.morton_to_cartesian(Val(2), i) #[cite: 1]

                # Compute physical coordinates for the bottom-left corner
                x_bl = min_corner[1] + (cartesian[1] - 1) * finest_cell_length
                y_bl = min_corner[2] + (cartesian[2] - 1) * finest_cell_length

                # Draw the cell bounding box
                rect = Shape([x_bl, x_bl + cell_size, x_bl + cell_size, x_bl],
                             [y_bl, y_bl, y_bl + cell_size, y_bl + cell_size])
                plot!(p, rect, fillalpha = 0.0, linecolor = :black, linewidth = 1.5,
                      label = "")

                # Annotate the cell index in the bottom-left corner
                annotate!(p, x_bl + 0.05 * cell_size, y_bl + 0.05 * cell_size,
                          text(string(i), 9, :blue, :bottom, :left))
            end
        end

        # Scatter the points
        scatter!(p, coords[1, :], coords[2, :],
                 markercolor = :red, markershape = :circle, markersize = 5,
                 label = "Particles")

        # Handle zooming and plot limits
        if zoom_to_points && !isempty(coords)
            min_x, max_x = minimum(coords[1, :]), maximum(coords[1, :])
            min_y, max_y = minimum(coords[2, :]), maximum(coords[2, :])

            # Ensure a minimum width/height for the zoom window even if points are aligned
            dx = max(max_x - min_x, finest_cell_length)
            dy = max(max_y - min_y, finest_cell_length)

            # Apply limits clamped to the grid boundaries so we don't zoom into empty void
            plot!(p,
                  xlims = (max(min_corner[1], min_x - zoom_padding * dx),
                           min(min_corner[1] + L_grid, max_x + zoom_padding * dx)),
                  ylims = (max(min_corner[2], min_y - zoom_padding * dy),
                           min(min_corner[2] + L_grid, max_y + zoom_padding * dy)))
        else
            # Default: show the full grid limits
            plot!(p, xlims = (min_corner[1], min_corner[1] + L_grid),
                  ylims = (min_corner[2], min_corner[2] + L_grid))
        end

        return p
    end
end;
