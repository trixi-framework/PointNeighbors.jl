@testset verbose=true "TreeNeighborhoodSearch" begin
    # Fixed seed to ensure reproducibility of the random point clouds below.
    Random.seed!(1)

    # Brute-force reference: all points `j` in `y` with `|x - y_j| <= h`.
    # With periodicity, use the distance to the closest periodic image.
    function brute_force_neighbors(x, h, y; eachindex_y = axes(y, 2),
                                   periodic_box = nothing)
        return [j
                for j in eachindex_y
                if sqrt(sum(abs2, periodic_difference(x - y[:, j], periodic_box))) <= h]
    end

    periodic_difference(pos_diff, ::Nothing) = pos_diff
    function periodic_difference(pos_diff, periodic_box)
        return pos_diff - periodic_box.size .* round.(pos_diff ./ periodic_box.size)
    end

    # All neighbors of the point `x` with search radius `h` found by the tree NHS, sorted.
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
                                 root_cell_size = 1.0)

        error_str = "a 3D cell list is required for a TreeNeighborhoodSearch{3}"
        @test_throws error_str TreeNeighborhoodSearch{3}(; cell_list)

        error_str = "`TreeNeighborhoodSearch` requires a `TreeCellList`"
        @test_throws error_str TreeNeighborhoodSearch{2}(cell_list = DictionaryCellList{2}())

        error_str = "is not a valid update strategy"
        @test_throws error_str TreeNeighborhoodSearch{2}(; cell_list,
                                                         update_strategy = SerialIncrementalUpdate())

        error_str = "the search radius must not exceed the `root_cell_size`"
        @test_throws error_str TreeNeighborhoodSearch{2}(; cell_list, search_radius = 1.1)

        error_str = "the search radius must not be negative or NaN"
        @test_throws error_str TreeNeighborhoodSearch{2}(; cell_list, search_radius = -0.1)
        @test_throws error_str TreeNeighborhoodSearch{2}(; cell_list, search_radius = NaN)

        error_str = "`root_cell_size` must not be negative or NaN"
        @test_throws error_str TreeCellList(min_corner = (0.0, 0.0),
                                            max_corner = (1.0, 1.0),
                                            root_cell_size = -0.1)
        @test_throws error_str TreeCellList(min_corner = (0.0, 0.0),
                                            max_corner = (1.0, 1.0),
                                            root_cell_size = NaN)

        # The root cell size is required unless creating a template.
        template_cell_list = TreeCellList(min_corner = (0.0, 0.0), max_corner = (1.0, 1.0))
        error_str = "the `root_cell_size` of the `TreeCellList` must be set"
        @test_throws error_str TreeNeighborhoodSearch{2}(cell_list = template_cell_list,
                                                         search_radius = 0.1)

        # The element type is determined by the domain corners.
        cell_list = TreeCellList(min_corner = (0.0f0, 0.0f0), max_corner = (1.0f0, 2.0f0),
                                 root_cell_size = 0.25f0)
        nhs = TreeNeighborhoodSearch{2}(; cell_list, search_radius = 0.1)
        @test eltype(nhs) == Float32
        @test nhs.search_radius isa Float32

        # A grid of 4 x 8 root cells covering the domain,
        # with one layer of root cells around it
        @test nhs.cell_list.n_roots == (6, 10)
        @test nhs.cell_list.grid_min_corner == [-0.25f0, -0.25f0]

        # With periodicity, the element type is also determined by the cell list
        # when the periodic box has a different element type.
        periodic_box = PeriodicBox(min_corner = [0.0, 0.0], max_corner = [1.0, 2.0])
        nhs = TreeNeighborhoodSearch{2}(; cell_list, search_radius = 0.1, periodic_box)
        @test eltype(nhs) == Float32
        @test nhs.cell_list.grid_min_corner isa SVector{2, Float32}
        @test nhs.cell_list.actual_root_cell_size isa SVector{2, Float32}

        # A periodic box with integer corners.
        cell_list = TreeCellList(min_corner = (0.0, 0.0), max_corner = (1.0, 2.0),
                                 root_cell_size = 0.25)
        periodic_box = PeriodicBox(min_corner = [0, 0], max_corner = [1, 2])
        nhs = TreeNeighborhoodSearch{2}(; cell_list, search_radius = 0.1, periodic_box)
        @test eltype(nhs) == Float64
        @test nhs.cell_list.grid_min_corner == [0.0, 0.0]
        @test nhs.cell_list.actual_root_cell_size == [0.25, 0.25]
    end

    @testset "Leaf Level" begin
        cell_list = TreeCellList(min_corner = (0.0, 0.0), max_corner = (1.0, 1.0),
                                 root_cell_size = 1.0)
        function leaf_level(search_radius)
            nhs = TreeNeighborhoodSearch{2}(; cell_list, search_radius)
            return nhs.cell_list.leaf_level
        end

        # The leaves are at the finest level with cells that are at least as large
        # as the search radius.
        @test leaf_level(1.0) == 0
        @test leaf_level(0.3) == 1
        @test leaf_level(0.25) == 2
        @test leaf_level(0.2) == 2
        @test leaf_level(0.01) == 6

        # One list is allocated for each leaf of the 3 x 3 root cells.
        nhs = TreeNeighborhoodSearch{2}(; cell_list, search_radius = 0.3)
        @test length(nhs.cell_list.cells) == 9 * 4

        # With periodicity, the root cells are stretched differently in each dimension.
        # The periodic box of size 0.95 x 0.75 is tiled by 3 x 3 root cells of size
        # 0.95 / 3 ≈ 0.317 x 0.75 / 3 = 0.25. The leaf level must be chosen from the
        # smallest root cell size. Level 1 would yield leaves of size 0.25 / 2 = 0.125
        # in y-direction, which is smaller than the search radius 0.14.
        # In x-direction, leaves of size 0.317 / 2 ≈ 0.158 would be large enough.
        periodic_box = PeriodicBox(min_corner = [0.0, 0.0], max_corner = [0.95, 0.75])
        cell_list = TreeCellList(min_corner = (0.0, 0.0), max_corner = (0.95, 0.75),
                                 root_cell_size = 0.25)
        search_radius = 0.14
        nhs = TreeNeighborhoodSearch{2}(; cell_list, search_radius, periodic_box)
        @test nhs.cell_list.n_roots == (3, 3)
        @test nhs.cell_list.leaf_level == 0

        coords = periodic_box.size .* rand(2, 200)
        initialize!(nhs, coords, coords)
        @test all(axes(coords, 2)) do i
            tree_neighbors(nhs, coords[:, i], search_radius, coords) ==
            brute_force_neighbors(coords[:, i], search_radius, coords; periodic_box)
        end
    end

    @testset "Spread Bits $(NDIMS)D" for NDIMS in 2:3
        # Reference implementation of `spread_bits` moving one bit at a time.
        # Only the lower 32 bits in 2D and 21 bits in 3D are used, so that the result
        # fits into the lower 64 bits.
        n_bits = (32, 21)[NDIMS - 1]
        function spread_bits_reference(x)
            result = 0
            for i in 0:(n_bits - 1)
                result |= ((x >> i) & 1) << (NDIMS * i)
            end
            return result
        end

        max_value = (1 << n_bits) - 1
        single_bits = [1 << i for i in 0:(n_bits - 1)]
        # Alternating bit patterns 0b...0101 and 0b...1010
        alternating = [0x5555555555555555 % Int & max_value,
            0xaaaaaaaaaaaaaaaa % Int & max_value]
        inputs = [0; 1; max_value; single_bits; alternating; rand(0:max_value, 1000)]

        @test all(inputs) do x
            PointNeighbors.spread_bits(x, Val(NDIMS)) == spread_bits_reference(x)
        end
    end

    @testset "Morton Order $(NDIMS)D" for NDIMS in 2:3
        # A grid of 3 x 2 (x 2) root cells.
        cell_list = TreeCellList(min_corner = zeros(NDIMS),
                                 max_corner = [3.0, 2.0, 2.0][1:NDIMS],
                                 root_cell_size = 1.0)

        for level in 0:4
            n_cells = PointNeighbors.n_cells_per_dimension(cell_list, level)
            cells = [Tuple(cell) .- 1 for cell in CartesianIndices(n_cells)]
            indices = [PointNeighbors.cell_index(cell_list, cell, level) for cell in cells]

            # Each cell has a unique zero-based index.
            @test sort(vec(indices)) == 0:(prod(n_cells) - 1)

            # The parent of a cell is obtained by removing the last `NDIMS` bits
            # of the index.
            if level > 0
                @test all(zip(cells, indices)) do (cell, index)
                    parent = PointNeighbors.cell_index(cell_list, cell .>> 1, level - 1)
                    return index >> NDIMS == parent
                end
            end
        end
    end

    @testset "Z-Order" begin
        # A 2D grid of 3 x 2 root cells.
        cell_list = TreeCellList(min_corner = (0.0, 0.0), max_corner = (3.0, 2.0),
                                 root_cell_size = 1.0)

        # Within a root cell, the cells are sorted in Z-order.
        z_order = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (3, 0), (2, 1), (3, 1)]
        @test [PointNeighbors.cell_index(cell_list, cell, 2) for cell in z_order] == 0:7

        # The second root cell in x-direction starts after the 16 cells of the first one.
        @test PointNeighbors.cell_index(cell_list, (4, 0), 2) == 16
    end

    @testset "Neighbor Index Parts $(NDIMS)D" for NDIMS in 1:3
        # The parts of the neighboring leaves computed with Morton increments and decrements
        # must match the parts computed with `cell_index_part`, also when the neighboring
        # leaf is in the neighboring root cell.
        cell_list_ = TreeCellList(min_corner = zeros(NDIMS), max_corner = fill(3.0, NDIMS),
                                  root_cell_size = 1.0)
        nhs = TreeNeighborhoodSearch{NDIMS}(cell_list = cell_list_, search_radius = 0.25)
        (; cell_list) = nhs
        n_leaves = PointNeighbors.n_cells_per_dimension(cell_list, 2)

        for dim in 1:NDIMS, leaf in 1:(n_leaves[dim] - 2)
            parts = PointNeighbors.neighbor_index_parts(cell_list, dim, leaf, nothing)
            expected = Tuple(PointNeighbors.cell_index_part(cell_list, dim, leaf + i, 2)
                             for i in -1:1)
            @test parts == expected
        end

        # With periodicity, the neighboring leaves of the leaves at the boundary are
        # the leaves at the opposite boundary.
        periodic_box = PeriodicBox(min_corner = zeros(NDIMS), max_corner = fill(3.0, NDIMS))
        cell_list_ = TreeCellList(min_corner = zeros(NDIMS), max_corner = fill(3.0, NDIMS),
                                  root_cell_size = 1.0)
        nhs = TreeNeighborhoodSearch{NDIMS}(; cell_list = cell_list_,
                                            search_radius = 0.25, periodic_box)
        (; cell_list) = nhs
        n_leaves = PointNeighbors.n_cells_per_dimension(cell_list, 2)

        for dim in 1:NDIMS, leaf in 0:(n_leaves[dim] - 1)
            parts = PointNeighbors.neighbor_index_parts(cell_list, dim, leaf, periodic_box)
            expected = Tuple(PointNeighbors.cell_index_part(cell_list, dim,
                                                            mod(leaf + i, n_leaves[dim]),
                                                            2)
                             for i in -1:1)
            @test parts == expected
        end
    end

    @testset verbose=true "Compare Against Brute Force $(NDIMS)D" for NDIMS in 1:3
        n_points = (20, 200, 2_000)[NDIMS]
        coords = rand(NDIMS, n_points)

        # Move the first point to the center of the unit cube, which is on a leaf boundary
        # for most layouts below (but not, e.g., with root cells of size 0.3),
        # and the second point to the corner of the domain.
        coords[:, 1] .= 0.5
        coords[:, 2] .= 1.0

        # Different tree configurations
        min_corner = zeros(NDIMS)
        max_corner = ones(NDIMS)
        cell_lists = [
            "Root Cells of Size 0.125" => TreeCellList(; min_corner, max_corner,
                                                       root_cell_size = 0.125),
            "Vector{Vector} Backend" => TreeCellList(; min_corner, max_corner,
                                                     root_cell_size = 0.125,
                                                     backend = Vector{Vector{Int32}}),
            "Root Cells of Size 1" => TreeCellList(; min_corner, max_corner,
                                                   root_cell_size = 1.0),
            "Root Cells Not Aligned With Domain" => TreeCellList(; min_corner,
                                                                 max_corner,
                                                                 root_cell_size = 0.3),
            # With search radius 0.125, the leaves are the root cells of size 0.2,
            # which are larger than the search radius.
            "Large Leaves" => TreeCellList(; min_corner, max_corner,
                                           root_cell_size = 0.2),
            # A domain that is larger than the point cloud
            "Large Domain" => TreeCellList(; min_corner = fill(-1.0, NDIMS),
                                           max_corner = fill(2.0, NDIMS),
                                           root_cell_size = 0.5),
            # A domain with a different number of root cells in each dimension.
            "Non-Cubic Domain" => TreeCellList(; min_corner = [-0.2, 0.0, -0.4][1:NDIMS],
                                               max_corner = [1.0, 1.6, 1.35][1:NDIMS],
                                               root_cell_size = 0.25)
        ]

        # Search radii for which the leaves are exactly as large as the search radius
        # with some of the layouts above and larger than the search radius with the others.
        @testset "$cell_list_name, Search Radius $search_radius" for (cell_list_name,
                                                                      cell_list) in
                                                                     cell_lists,
                                                                     search_radius in
                                                                     (0.125, 0.1)

            nhs = TreeNeighborhoodSearch{NDIMS}(; cell_list, search_radius)

            # Initialize with different coordinates first to also test the update.
            initialize!(nhs, coords, rand(NDIMS, n_points))
            update!(nhs, coords, coords)

            # Query each point of the tree
            @test all(axes(coords, 2)) do i
                tree_neighbors(nhs, coords[:, i], search_radius, coords) ==
                brute_force_neighbors(coords[:, i], search_radius, coords)
            end

            # Query arbitrary points, including points outside of the domain,
            # with query radii that are not larger than the search radius.
            query_coords = 1.6 .* rand(NDIMS, 200) .- 0.3
            query_radii = search_radius .* rand(200)
            @test all(axes(query_coords, 2)) do i
                x = query_coords[:, i]
                tree_neighbors(nhs, x, query_radii[i], coords) ==
                brute_force_neighbors(x, query_radii[i], coords)
            end
        end
    end

    @testset verbose=true "Periodicity $(NDIMS)D" for NDIMS in 1:3
        n_points = (20, 200, 2_000)[NDIMS]

        # A periodic box that is not a multiple of the root cell size in all dimensions.
        min_corner = fill(-0.1, NDIMS)
        max_corner = [0.9, 0.6, 0.85][1:NDIMS]
        periodic_box = PeriodicBox(; min_corner, max_corner)
        coords = min_corner .+ periodic_box.size .* rand(NDIMS, n_points)

        # Points on the boundary of the periodic box
        coords[:, 1] .= min_corner
        coords[:, 2] .= max_corner

        cell_lists = [
            "Root Cells of Size 0.1" => TreeCellList(; min_corner, max_corner,
                                                     root_cell_size = 0.1),
            "Root Cells of Size 0.3" => TreeCellList(; min_corner, max_corner,
                                                     root_cell_size = 0.3)
        ]

        @testset "$cell_list_name" for (cell_list_name, cell_list) in cell_lists
            search_radius = 0.1
            nhs = TreeNeighborhoodSearch{NDIMS}(; cell_list, search_radius, periodic_box)
            initialize!(nhs, coords, coords)

            @test all(axes(coords, 2)) do i
                tree_neighbors(nhs, coords[:, i], search_radius, coords) ==
                brute_force_neighbors(coords[:, i], search_radius, coords;
                                      periodic_box)
            end

            # Query arbitrary points, including points outside of the periodic box.
            query_coords = min_corner .- 0.5 .+ (periodic_box.size .+ 1) .* rand(NDIMS, 200)
            @test all(axes(query_coords, 2)) do i
                x = query_coords[:, i]
                tree_neighbors(nhs, x, search_radius, coords) ==
                brute_force_neighbors(x, search_radius, coords; periodic_box)
            end
        end
    end

    @testset "Inactive Points" begin
        coords = rand(3, 100)

        cell_list = TreeCellList(min_corner = (0.0, 0.0, 0.0),
                                 max_corner = (1.0, 1.0, 1.0), root_cell_size = 0.5)
        nhs = TreeNeighborhoodSearch{3}(; cell_list, search_radius = 0.1)

        # Only every third point is part of the tree.
        eachindex_y = 1:3:100
        initialize!(nhs, coords, coords; eachindex_y)

        @test all(axes(coords, 2)) do i
            tree_neighbors(nhs, coords[:, i], 0.1, coords) ==
            brute_force_neighbors(coords[:, i], 0.1, coords; eachindex_y)
        end
    end

    @testset "`copy_neighborhood_search`" begin
        cell_list = TreeCellList(min_corner = (0.0, 0.0), max_corner = (1.0, 2.0),
                                 root_cell_size = 0.5)
        nhs = TreeNeighborhoodSearch{2}(; cell_list, update_strategy = SerialUpdate())

        copy = copy_neighborhood_search(nhs, 0.1, 10)

        @test copy.search_radius == 0.1
        @test copy.cell_list.min_corner == cell_list.min_corner
        @test copy.cell_list.max_corner == cell_list.max_corner
        @test copy.cell_list.cells isa PointNeighbors.DynamicVectorOfVectors
        @test copy.update_strategy == SerialUpdate()

        # The root cells of the copy have the size of the search radius, so the leaves are
        # the root cells. There are 10 x 20 root cells and one layer of root cells around
        # the domain.
        @test copy.cell_list.root_cell_size == 0.1
        @test copy.cell_list.leaf_level == 0
        @test copy.cell_list.n_roots == (12, 22)

        # The copy has its own lists.
        @test copy.cell_list.cells !== nhs.cell_list.cells

        # With periodicity.
        periodic_box = PeriodicBox(min_corner = [0.0, 0.0], max_corner = [1.0, 2.0])
        nhs = TreeNeighborhoodSearch{2}(; cell_list, periodic_box)
        copy = copy_neighborhood_search(nhs, 0.1, 10)

        @test copy.periodic_box === periodic_box
        @test copy.cell_list.n_roots == (10, 20)
    end

    @testset "Errors" begin
        cell_list = TreeCellList(min_corner = (0.0, 0.0), max_corner = (1.0, 1.0),
                                 root_cell_size = 1.0)
        nhs = TreeNeighborhoodSearch{2}(; cell_list, search_radius = 0.1)

        error_str = "particle coordinates are NaN or outside the domain bounds"
        @test_throws error_str initialize!(nhs, [0.1 1.5; 0.2 0.5], [0.1 1.5; 0.2 0.5])
        @test_throws error_str initialize!(nhs, [0.1 NaN; 0.2 0.5], [0.1 NaN; 0.2 0.5])

        # The domain must be the periodic box.
        periodic_box = PeriodicBox(min_corner = [0.0, 0.0], max_corner = [2.0, 1.0])
        error_str = "the domain of the `TreeCellList` must be the periodic box"
        @test_throws error_str TreeNeighborhoodSearch{2}(; cell_list, periodic_box,
                                                         search_radius = 0.1)

        # At least 3 leaves in each dimension with periodicity.
        periodic_box = PeriodicBox(min_corner = [0.0, 0.0], max_corner = [1.0, 1.0])
        error_str = "needs at least 3 leaves in each dimension when used with periodicity"
        @test_throws error_str TreeNeighborhoodSearch{2}(; cell_list, periodic_box,
                                                         search_radius = 0.4)
    end
end;
