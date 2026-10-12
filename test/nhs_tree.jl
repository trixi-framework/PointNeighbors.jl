@testset verbose=true "TreeNeighborhoodSearch" begin
    # Fixed seed to ensure reproducibility of the random point clouds below.
    Random.seed!(1)

    # Search radius of a pair of a query point with radius `h` and a point with radius `h_j`
    reference_radius(::MaximumRadius, h, h_j) = max(h, h_j)
    reference_radius(::AverageRadius, h, h_j) = (h + h_j) / 2
    reference_radius(::QueryRadius, h, h_j) = h

    # Brute-force reference: all points `j` in `y` with `|x - y_j| <= r(h, h_j)`.
    # With periodicity, use the distance to the closest periodic image.
    function brute_force_neighbors(x, h, y, radii; eachindex_y = axes(y, 2),
                                   neighbor_criterion = MaximumRadius(),
                                   periodic_box = nothing)
        # Compute all distances at once, which is much faster than a loop over all points
        # with a closure inside of a test set
        pos_diff = periodic_difference(x .- y, periodic_box)
        distances = vec(sqrt.(sum(abs2, pos_diff, dims = 1)))

        return [j
                for j in eachindex_y
                if distances[j] <= reference_radius(neighbor_criterion, h, radii[j])]
    end

    periodic_difference(pos_diff, ::Nothing) = pos_diff
    function periodic_difference(pos_diff, periodic_box)
        return pos_diff .- periodic_box.size .* round.(pos_diff ./ periodic_box.size)
    end

    neighbor_criteria = (MaximumRadius(), AverageRadius(), QueryRadius())

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

    # Check that the tree NHS finds the same neighbors as the brute-force reference
    # for all query points `query_coords` with search radii `query_radii`.
    # This function barrier avoids slow closures over variables of the test sets.
    function neighbors_match(nhs, query_coords, query_radii, y, radii; kwargs...)
        return all(axes(query_coords, 2)) do i
            x = query_coords[:, i]
            tree_neighbors(nhs, x, query_radii[i], y) ==
            brute_force_neighbors(x, query_radii[i], y, radii; kwargs...)
        end
    end

    @testset "Constructor" begin
        cell_list = TreeCellList(min_corner = (0.0, 0.0), max_corner = (1.0, 1.0),
                                 root_cell_size = 1.0)

        error_str = "a 3D cell list is required for a TreeNeighborhoodSearch{3}"
        @test_throws error_str TreeNeighborhoodSearch{3}(; cell_list)

        error_str = "`TreeNeighborhoodSearch` requires a `TreeCellList`"
        @test_throws error_str TreeNeighborhoodSearch{2}(cell_list = DictionaryCellList{2}())

        error_str = "`neighbor_criterion` must be"
        @test_throws error_str TreeNeighborhoodSearch{2}(; cell_list,
                                                         neighbor_criterion = :maximum)

        # The maximum radius is the default criterion
        @test TreeNeighborhoodSearch{2}(; cell_list).neighbor_criterion == MaximumRadius()

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

        error_str = "`max_level` must be between 0 and 20 in 3D"
        @test_throws error_str TreeCellList(min_corner = (0.0, 0.0, 0.0),
                                            max_corner = (1.0, 1.0, 1.0),
                                            root_cell_size = 1.0, max_level = 21)

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

            # The coordinates can be recovered from the index.
            @test all(zip(cells, indices)) do (cell, index)
                PointNeighbors.cell_coords_from_index(cell_list, index, level) == cell
            end

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
        # The parts of the neighboring cells computed with Morton increments and decrements
        # must match the parts computed with `cell_index_part`, also when the neighboring
        # cell is in the neighboring root cell.
        cell_list = TreeCellList(min_corner = zeros(NDIMS), max_corner = fill(3.0, NDIMS),
                                 root_cell_size = 1.0)

        for level in 0:3
            n_cells = PointNeighbors.n_cells_per_dimension(cell_list, level)

            for dim in 1:NDIMS, cell in 1:(n_cells[dim] - 2)
                parts = PointNeighbors.neighbor_index_parts(cell_list, dim, cell, level,
                                                            nothing)
                expected = Tuple(PointNeighbors.cell_index_part(cell_list, dim, cell + i,
                                                                level)
                                 for i in -1:1)
                @test parts == expected
            end
        end

        # With periodicity, the neighboring cells of the cells at the boundary are
        # the cells at the opposite boundary.
        periodic_box = PeriodicBox(min_corner = zeros(NDIMS), max_corner = fill(3.0, NDIMS))
        cell_list_ = TreeCellList(min_corner = zeros(NDIMS), max_corner = fill(3.0, NDIMS),
                                  root_cell_size = 1.0, max_level = 3)
        nhs = TreeNeighborhoodSearch{NDIMS}(; cell_list = cell_list_, periodic_box)
        (; cell_list) = nhs

        for level in 0:3
            n_cells = PointNeighbors.n_cells_per_dimension(cell_list, level)

            for dim in 1:NDIMS, cell in 0:(n_cells[dim] - 1)
                parts = PointNeighbors.neighbor_index_parts(cell_list, dim, cell, level,
                                                            periodic_box)
                expected = Tuple(PointNeighbors.cell_index_part(cell_list, dim,
                                                                mod(cell + i, n_cells[dim]),
                                                                level)
                                 for i in -1:1)
                @test parts == expected
            end
        end
    end

    @testset "Uniform Search Radius" begin
        cell_list = TreeCellList(min_corner = (0.0, 0.0), max_corner = (1.0, 1.0),
                                 root_cell_size = 1.0)
        function max_level(search_radius)
            nhs = TreeNeighborhoodSearch{2}(; cell_list, search_radius)
            return nhs.cell_list.max_level
        end

        # By default, the finest level is the finest level with cells that are at least
        # as large as the search radius
        @test max_level(1.0) == 0
        @test max_level(0.3) == 1
        @test max_level(0.25) == 2
        @test max_level(0.2) == 2
        @test max_level(0.01) == 6

        # With a uniform search radius, all leaves containing points are at this level
        coords = rand(2, 500)
        nhs = TreeNeighborhoodSearch{2}(; cell_list, search_radius = 0.1, n_points = 500)
        initialize!(nhs, coords, coords)

        @test nhs.cell_list.max_level == 3
        @test all(axes(coords, 2)) do point
            finest_cell = PointNeighbors.finest_cell_index(SVector{2}(coords[:, point]),
                                                           nhs.cell_list, nothing)
            PointNeighbors.leaf_level(nhs.cell_list, finest_cell) == 3
        end

        # With periodicity, the root cells are stretched differently in each dimension.
        # The periodic box of size 0.95 x 0.75 is tiled by 3 x 3 root cells of size
        # 0.95 / 3 ≈ 0.317 x 0.75 / 3 = 0.25. The finest level must be chosen from the
        # smallest root cell size. Level 1 would yield cells of size 0.25 / 2 = 0.125
        # in y-direction, which is smaller than the search radius 0.14.
        # In x-direction, cells of size 0.317 / 2 ≈ 0.158 would be large enough.
        periodic_box = PeriodicBox(min_corner = [0.0, 0.0], max_corner = [0.95, 0.75])
        cell_list = TreeCellList(min_corner = (0.0, 0.0), max_corner = (0.95, 0.75),
                                 root_cell_size = 0.25)
        search_radius = 0.14
        nhs = TreeNeighborhoodSearch{2}(; cell_list, search_radius, periodic_box,
                                        n_points = 200)
        @test nhs.cell_list.n_roots == (3, 3)
        @test nhs.cell_list.max_level == 0

        coords = periodic_box.size .* rand(2, 200)
        initialize!(nhs, coords, coords)
        radii = fill(search_radius, size(coords, 2))
        @test neighbors_match(nhs, coords, radii, coords, radii; periodic_box)
    end

    @testset "Tree Structure" begin
        # In 1D, a tree with 3 levels below the root cell [0, 1) has cells of size
        # 1/2, 1/4 and 1/8. We look at the level of the leaf containing each of the
        # 8 finest cells of this root cell, which is the second root cell after the
        # padding root cell [-1, 0).
        cell_list = TreeCellList(min_corner = (0.0,), max_corner = (1.0,),
                                 root_cell_size = 1.0, max_level = 3)
        nhs = TreeNeighborhoodSearch{1}(; cell_list, n_points = 2)
        finest_leaf_levels(nhs) = [PointNeighbors.leaf_level(nhs.cell_list, cell)
                                   for cell in 8:15]

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
        @test finest_leaf_levels(nhs) == [3, 3, 2, 2, 2, 2, 2, 2]

        # The empty padding root cells are not split
        @test PointNeighbors.leaf_level(nhs.cell_list, 0) == 0
        @test PointNeighbors.leaf_level(nhs.cell_list, 16) == 0

        # Now increase the radius of the right point to 0.4.
        # The cell [0, 0.5) is not split anymore, although it only contains a point with
        # a small radius. The large radius in the neighboring cell [0.5, 1) could
        # otherwise reach beyond the 3^d block around the leaf containing the left point.
        radii = [0.1, 0.4]
        update_tree!(nhs, coords, radii)
        @test finest_leaf_levels(nhs) == [1, 1, 1, 1, 1, 1, 1, 1]

        # With a capacity of 1 point per cell, only the root cell with 2 points is split
        cell_list = TreeCellList(min_corner = (0.0,), max_corner = (1.0,),
                                 root_cell_size = 1.0, max_level = 3, capacity = 1)
        nhs = TreeNeighborhoodSearch{1}(; cell_list, n_points = 2)
        initialize_tree!(nhs, coords, [0.1, 0.2])
        @test finest_leaf_levels(nhs) == [1, 1, 1, 1, 1, 1, 1, 1]
    end

    @testset "Pruning of Cells Out of Reach" begin
        # A single point in the cell [0.5, 1) x [0, 0.5) at level 1 with radius 0.2
        cell_list = TreeCellList(min_corner = (0.0, 0.0), max_corner = (1.0, 1.0),
                                 root_cell_size = 1.0, max_level = 2)

        # Check if the cell `cell` at level `level` is out of reach for a query point
        # with coordinates `x` and search radius `h`.
        # The cell coordinates are relative to the domain. The grid of root cells starts
        # with a padding root cell, which has `2^level` cells at level `level`.
        function is_out_of_reach(nhs, cell, level, x, h)
            grid_cell = cell .+ (1 << level)
            cell_index = PointNeighbors.cell_index(nhs.cell_list, grid_cell, level)
            max_pair_radius = PointNeighbors.cell_max_pair_radius(nhs, grid_cell,
                                                                  cell_index, level,
                                                                  SVector(x...), h,
                                                                  max(h, 0.2))
            return max_pair_radius < 0
        end

        @testset "AverageRadius" begin
            nhs = TreeNeighborhoodSearch{2}(; cell_list, n_points = 1, pruning = true,
                                            neighbor_criterion = AverageRadius())
            initialize_tree!(nhs, [0.9; 0.1;;], [0.2])

            # The query point (0.2, 0.25) has a distance of 0.3 to this cell.
            # With a query radius of 0.4, points in this cell with a radius of 0.2 can be
            # at a distance of up to (0.4 + 0.2) / 2 = 0.3 to the query point.
            @test !is_out_of_reach(nhs, (1, 0), 1, (0.2, 0.25), 0.4)

            # With a query radius of 0.3, this is only (0.3 + 0.2) / 2 = 0.25
            @test is_out_of_reach(nhs, (1, 0), 1, (0.2, 0.25), 0.3)

            # The same holds for the leaf [0.75, 1) x [0, 0.25) containing the point
            # at level 2
            @test !is_out_of_reach(nhs, (3, 0), 2, (0.45, 0.1), 0.4)
            @test is_out_of_reach(nhs, (3, 0), 2, (0.45, 0.1), 0.2)

            # Empty cells are out of reach when their distance exceeds half the query
            # radius
            @test !is_out_of_reach(nhs, (0, 1), 1, (0.2, 0.25), 0.5)
            @test is_out_of_reach(nhs, (0, 1), 1, (0.2, 0.25), 0.49)
        end

        @testset "MaximumRadius" begin
            nhs = TreeNeighborhoodSearch{2}(; cell_list, n_points = 1, pruning = true)
            initialize_tree!(nhs, [0.9; 0.1;;], [0.2])

            # With a query radius of 0.3, points in this cell can be at a distance of
            # up to max(0.3, 0.2) = 0.3 to the query point
            @test !is_out_of_reach(nhs, (1, 0), 1, (0.2, 0.25), 0.3)

            # With a query radius of 0.25, this is only max(0.25, 0.2) = 0.25
            @test is_out_of_reach(nhs, (1, 0), 1, (0.2, 0.25), 0.25)

            # Points in this cell with a radius of 0.2 reach a distance of 0.2,
            # even for small query radii
            @test !is_out_of_reach(nhs, (3, 0), 2, (0.55, 0.1), 0.01)
            @test is_out_of_reach(nhs, (3, 0), 2, (0.54, 0.1), 0.01)

            # Empty cells are out of reach when their distance exceeds the query radius
            @test !is_out_of_reach(nhs, (0, 1), 1, (0.2, 0.25), 0.25)
            @test is_out_of_reach(nhs, (0, 1), 1, (0.2, 0.25), 0.24)
        end
    end

    @testset "Points Stored in Leaves" begin
        # Random points in 2D with radii growing in x-direction
        coords = rand(2, 500)
        radii = 0.01 .+ 0.1 .* coords[1, :]

        cell_list_ = TreeCellList(min_corner = (0.0, 0.0), max_corner = (1.0, 1.0),
                                  root_cell_size = 1.0, max_level = 6)
        nhs = TreeNeighborhoodSearch{2}(; cell_list = cell_list_, n_points = 500)
        (; cell_list) = nhs

        # Leave out some points
        eachindex_y = 1:2:500
        initialize_tree!(nhs, coords, radii; eachindex_y)

        # Loop over all leaves of the 3 x 3 root cells by jumping from leaf to leaf over
        # the finest cells. The finest cells of each leaf are contiguous in Morton order.
        n_finest_cells = 9 * 4^6
        cell = 0
        points_found = Int[]
        while cell < n_finest_cells
            level = PointNeighbors.leaf_level(cell_list, cell)
            next_cell = cell + PointNeighbors.n_finest_cells(cell_list, level)

            # The points of a leaf are stored in the list assigned to its first finest cell
            points = PointNeighbors.points_in_leaf(cell_list, cell)
            for point in points
                # The point must be inside this leaf. The grid of root cells starts
                # one root cell before the domain.
                cell_size = 1 / 2^level
                leaf_cell = floor.(Int, (coords[:, point] .+ 1) ./ cell_size)
                @test PointNeighbors.cell_index(cell_list, Tuple(leaf_cell), level) ==
                      cell >> (2 * (6 - level))
            end
            append!(points_found, points)

            # Empty leaves and all other finest cells of this leaf use the empty list 1
            if isempty(points)
                @test PointNeighbors.leaf_slot(cell_list, cell) == 1
            end
            @test all(i -> PointNeighbors.leaf_slot(cell_list, i) == 1,
                      (cell + 1):(next_cell - 1))

            cell = next_cell
        end

        # Each active point is stored in exactly one leaf
        @test sort(points_found) == eachindex_y

        # Lists are only allocated for the possible non-empty leaves and the empty list
        @test length(cell_list.cells) == 500 + 1
        @test isempty(cell_list.cells[1])
    end

    @testset verbose=true "Compare Against Brute Force $(NDIMS)D" for NDIMS in 1:3
        n_points = (20, 200, 2_000)[NDIMS]
        coords = rand(NDIMS, n_points)

        # Move the first point to the center of the unit cube, which is on a cell boundary
        # for most layouts below (but not, e.g., with root cells of size 0.3),
        # and the second point to the corner of the domain.
        center = fill(0.5, NDIMS)
        coords[:, 1] .= center
        coords[:, 2] .= 1.0

        # Different distributions of search radii in the unit cube
        radius_distributions = [
            "Smooth" => x -> 0.01 + 0.1 * x[1]^2,
            # A jump by a factor of 10 at a cell boundary, which violates any 2:1 balance
            # of the tree. Fine leaves on one side are next to coarse leaves on the other.
            "Jump" => x -> x[NDIMS] < 0.5 ? 0.02 : 0.2,
            # A single point with a large radius in the center surrounded by small radii.
            # Only the leaves around this point must be coarse.
            "Isolated Large Radius" => x -> x == center ? 0.25 : 0.02,
            # Arbitrary radii without any spatial correlation
            "Random" => x -> 0.01 + 0.2 * rand(),
            "Uniform" => x -> 0.05
        ]

        # Different tree configurations. The root cells must be at least as large as all
        # radii. Leaves are at most as large as the largest radius 0.25.
        min_corner = zeros(NDIMS)
        max_corner = ones(NDIMS)
        max_level = (6, 5, 3)[NDIMS]
        cell_lists = [
            "Root Cells of Size 0.25" => TreeCellList(; min_corner, max_corner,
                                                      root_cell_size = 0.25, max_level),
            "Vector{Vector} Backend" => TreeCellList(; min_corner, max_corner,
                                                     root_cell_size = 0.25, max_level,
                                                     backend = Vector{Vector{Int32}}),
            "Capacity 8" => TreeCellList(; min_corner, max_corner, root_cell_size = 0.25,
                                         max_level, capacity = 8),
            # Larger root cells, so that the 3^d block around a cell can be in multiple
            # root cells on every level
            "Root Cells of Size 1" => TreeCellList(; min_corner, max_corner,
                                                   root_cell_size = 1.0,
                                                   max_level = max_level + 2),
            "Root Cells Not Aligned With Domain" => TreeCellList(; min_corner,
                                                                 max_corner,
                                                                 root_cell_size = 0.3,
                                                                 max_level),
            # A domain that is larger than the point cloud
            "Large Domain" => TreeCellList(; min_corner = fill(-1.0, NDIMS),
                                           max_corner = fill(2.0, NDIMS),
                                           root_cell_size = 0.5,
                                           max_level = max_level + 1),
            # A domain with a different number of root cells in each dimension.
            "Non-Cubic Domain" => TreeCellList(; min_corner = [-0.2, 0.0, -0.4][1:NDIMS],
                                               max_corner = [1.0, 1.6, 1.35][1:NDIMS],
                                               root_cell_size = 0.25, max_level)
        ]

        @testset "$cell_list_name, $radii_name Radii" for (cell_list_name, cell_list) in
                                                          cell_lists,
                                                          (radii_name, radius) in
                                                          radius_distributions

            radii = [radius(coords[:, j]) for j in axes(coords, 2)]

            # The same tree is used for all neighbor criteria, and the same queries must
            # yield the same neighbors with and without pruning
            @testset "$(nameof(typeof(neighbor_criterion))), Pruning = $pruning" for neighbor_criterion in
                                                                                     neighbor_criteria,
                                                                                     pruning in
                                                                                     (false,
                                                                                      true)

                nhs = TreeNeighborhoodSearch{NDIMS}(; cell_list, n_points,
                                                    neighbor_criterion,
                                                    pruning)

                # Initialize with different coordinates first to also test the update
                initialize_tree!(nhs, rand(NDIMS, n_points), radii)
                update_tree!(nhs, coords, radii)

                # Query each point of the tree with its own radius
                @test neighbors_match(nhs, coords, radii, coords, radii; neighbor_criterion)

                # Query arbitrary points with arbitrary radii up to the size of the root
                # cells, including points outside of the domain
                query_coords = 1.6 .* rand(NDIMS, 200) .- 0.3
                query_radii = cell_list.root_cell_size .* rand(200) .^ 2
                @test neighbors_match(nhs, query_coords, query_radii, coords, radii;
                                      neighbor_criterion)
            end
        end

        # A uniform search radius with `initialize!` and `update!`.
        # Search radii for which the leaves are exactly as large as the search radius
        # with some of the layouts above and larger than the search radius with the others.
        @testset "$cell_list_name, Uniform Search Radius $search_radius" for (cell_list_name,
                                                                              cell_list_) in
                                                                             cell_lists,
                                                                             search_radius in
                                                                             (0.125,
                                                                              0.1)

            # Without the explicit `max_level`, the finest level is chosen for the search
            # radius
            cell_list = TreeCellList(; cell_list_.min_corner, cell_list_.max_corner,
                                     cell_list_.root_cell_size, cell_list_.capacity,
                                     backend = typeof(cell_list_.cells))
            nhs = TreeNeighborhoodSearch{NDIMS}(; cell_list, search_radius, n_points)

            # Initialize with different coordinates first to also test the update.
            initialize!(nhs, coords, rand(NDIMS, n_points))
            update!(nhs, coords, coords)

            radii = fill(search_radius, n_points)
            @test neighbors_match(nhs, coords, radii, coords, radii)
        end
    end

    @testset verbose=true "Periodicity" begin
        @testset verbose=true "Hand-Made Examples" begin
            # The same examples as for the other neighborhood searches,
            # see `test/neighborhood_search.jl`. All points have a search radius of 0.1.
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
                # The root cells are not cubic in this example
                PeriodicBox(min_corner = [-0.1, -0.2], max_corner = [0.205, 0.43]),
                PeriodicBox(min_corner = [-0.1, -0.2, 0.05], max_corner = [0.2, 0.4, 0.35])
            ]

            @testset "$(names[i])" for i in eachindex(names)
                coords = coordinates[i]
                periodic_box = periodic_boxes[i]
                NDIMS = size(coords, 1)

                cell_list = TreeCellList(; periodic_box.min_corner, periodic_box.max_corner,
                                         root_cell_size = 0.1, max_level = 3)
                nhs = TreeNeighborhoodSearch{NDIMS}(; cell_list, periodic_box,
                                                    n_points = size(coords, 2))
                initialize_tree!(nhs, coords, fill(0.1, size(coords, 2)))

                @test [tree_neighbors(nhs, coords[:, i], 0.1, coords)
                       for i in axes(coords, 2)] == [[1, 3, 5], [2], [1, 3], [4], [1, 5]]
            end
        end

        @testset verbose=true "Compare Against Brute Force $(NDIMS)D" for NDIMS in 1:3
            # A periodic box that is not a multiple of the root cell size,
            # so that the root cells are not cubic
            min_corner = fill(-0.1, NDIMS)
            max_corner = fill(0.9, NDIMS)
            max_corner[end] = 0.55
            periodic_box = PeriodicBox(; min_corner, max_corner)

            n_points = (20, 200, 2_000)[NDIMS]
            coords = min_corner .+ periodic_box.size .* rand(NDIMS, n_points)

            # Points on the boundary of the periodic box
            coords[:, 1] .= min_corner
            coords[:, 2] .= max_corner

            # Radii jumping by a factor of 10 at the periodic boundary in x-direction,
            # where the points with small radii on the left are next to the points with
            # large radii on the right. All radii must be smaller than the root cells.
            radii = [coords[1, j] < 0.4 ? 0.01 : 0.1 for j in axes(coords, 2)]

            cell_list = TreeCellList(; min_corner, max_corner, root_cell_size = 0.11,
                                     max_level = 4, max_points_per_cell = 500)

            # The same queries must yield the same neighbors with and without pruning
            @testset "$(nameof(typeof(neighbor_criterion))), Pruning = $pruning" for neighbor_criterion in
                                                                                     neighbor_criteria,
                                                                                     pruning in
                                                                                     (false,
                                                                                      true)

                nhs = TreeNeighborhoodSearch{NDIMS}(; cell_list, periodic_box, n_points,
                                                    neighbor_criterion, pruning)
                initialize_tree!(nhs, coords, radii)

                # The root cells tile the periodic box exactly
                (; actual_root_cell_size, n_roots) = nhs.cell_list
                @test actual_root_cell_size .* n_roots ≈ periodic_box.size

                # Query each point of the tree with its own radius
                @test neighbors_match(nhs, coords, radii, coords, radii; neighbor_criterion,
                                      periodic_box)

                # Query arbitrary points, including points outside of the periodic box,
                # with arbitrary radii up to the size of the root cells
                query_coords = min_corner .- 0.5 .+
                               2 .* periodic_box.size .* rand(NDIMS, 200)
                query_radii = 0.11 .* rand(200)
                @test neighbors_match(nhs, query_coords, query_radii, coords, radii;
                                      neighbor_criterion, periodic_box)
            end

            # A uniform search radius with `initialize!`
            @testset "Uniform Search Radius, Root Cells of Size $root_cell_size" for root_cell_size in
                                                                                     (0.1,
                                                                                      0.2)
                cell_list = TreeCellList(; min_corner, max_corner, root_cell_size,
                                         max_points_per_cell = 500)
                nhs = TreeNeighborhoodSearch{NDIMS}(; cell_list, search_radius = 0.1,
                                                    n_points,
                                                    periodic_box)
                initialize!(nhs, coords, coords)

                radii_ = fill(0.1, n_points)
                @test neighbors_match(nhs, coords, radii_, coords, radii_; periodic_box)
            end
        end

        @testset "Coarse Leaves at the Periodic Boundary" begin
            # Many points on the left and only a few points in the strip x > 0.9
            # at the right side of the periodic box
            periodic_box = PeriodicBox(min_corner = [0.0, 0.0], max_corner = [1.0, 1.0])
            coords = hcat([0.9; 1;;] .* rand(2, 5_000),
                          [0.9; 0;;] .+ [0.1; 1;;] .* rand(2, 20))
            radii = fill(0.03, size(coords, 2))

            # With a capacity of 8, the root cells in the right strip with only a few points
            # are not split. The other root cells are split into cells of size 0.05.
            cell_list = TreeCellList(; periodic_box.min_corner, periodic_box.max_corner,
                                     root_cell_size = 0.1, max_level = 3, capacity = 8)
            nhs = TreeNeighborhoodSearch{2}(; cell_list, periodic_box,
                                            n_points = size(coords, 2))
            initialize_tree!(nhs, coords, radii)

            # The leaves on the left side of the periodic boundary are finer than the
            # leaves on the right side, so the 3^d block around a leaf on the left
            # contains coarser leaves on the other side of the periodic boundary.
            function leaf_level(x)
                finest_cell = PointNeighbors.finest_cell_index(SVector(x...),
                                                               nhs.cell_list, periodic_box)
                return PointNeighbors.leaf_level(nhs.cell_list, finest_cell)
            end
            @test leaf_level(coords[:, 1]) == 1
            @test leaf_level(coords[:, end]) == 0

            @test neighbors_match(nhs, coords, radii, coords, radii; periodic_box)
        end

        @testset "Errors" begin
            periodic_box = PeriodicBox(min_corner = [0.0, 0.0], max_corner = [1.0, 0.5])

            # The domain of the cell list must be the periodic box.
            cell_list = TreeCellList(min_corner = [0.0, 0.0], max_corner = [1.0, 1.0],
                                     root_cell_size = 0.1, max_level = 3)
            error_str = "the domain of the `TreeCellList` must be the periodic box"
            @test_throws error_str TreeNeighborhoodSearch{2}(; cell_list, periodic_box)

            # At least 3 root cells are required in each dimension.
            cell_list = TreeCellList(min_corner = [0.0, 0.0], max_corner = [1.0, 0.5],
                                     root_cell_size = 0.2, max_level = 3)
            error_str = "needs at least 3 root cells in each dimension"
            @test_throws error_str TreeNeighborhoodSearch{2}(; cell_list, periodic_box)
        end
    end

    @testset "Neighbor Loops" begin
        # Points in 2D with a smooth radius distribution
        coords = rand(2, 1000)
        radii = 0.02 .+ 0.1 .* coords[1, :] .* coords[2, :]

        cell_list = TreeCellList(min_corner = (0.0, 0.0), max_corner = (1.0, 1.0),
                                 root_cell_size = 0.25, max_level = 5)
        nhs = TreeNeighborhoodSearch{2}(; cell_list, n_points = size(coords, 2))
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
        coords = rand(3, 100)
        radii = 0.05 .+ 0.1 .* coords[3, :]

        cell_list = TreeCellList(min_corner = (0.0, 0.0, 0.0),
                                 max_corner = (1.0, 1.0, 1.0), root_cell_size = 0.25,
                                 max_level = 3)
        nhs = TreeNeighborhoodSearch{3}(; cell_list, n_points = size(coords, 2))

        # Only every third point is part of the tree.
        eachindex_y = 1:3:100
        initialize_tree!(nhs, coords, radii; eachindex_y)

        @test neighbors_match(nhs, coords, radii, coords, radii; eachindex_y)
    end

    @testset "`copy_neighborhood_search`" begin
        cell_list = TreeCellList(min_corner = (0.0, 0.0), max_corner = (1.0, 2.0),
                                 root_cell_size = 0.5, max_level = 4, capacity = 3)
        nhs = TreeNeighborhoodSearch{2}(; cell_list, n_points = 10,
                                        neighbor_criterion = QueryRadius(),
                                        update_strategy = SerialUpdate())

        copy = copy_neighborhood_search(nhs, 0.1, 10)

        @test copy.search_radius == 0.1
        @test copy.cell_list.min_corner == cell_list.min_corner
        @test copy.cell_list.max_corner == cell_list.max_corner
        @test copy.cell_list.max_level == 4
        @test copy.cell_list.capacity == 3
        @test copy.cell_list.cells isa PointNeighbors.DynamicVectorOfVectors
        @test PointNeighbors.uses_pruning(copy) == PointNeighbors.uses_pruning(nhs)
        @test copy.neighbor_criterion == QueryRadius()
        @test copy.update_strategy == SerialUpdate()

        # The root cells of the copy have the size of the search radius.
        # There are 10 x 20 root cells and one layer of root cells around the domain.
        @test copy.cell_list.root_cell_size == 0.1
        @test copy.cell_list.n_roots == (12, 22)

        # The copy has its own data structures
        @test copy.cell_list.leaf_level !== nhs.cell_list.leaf_level

        # Without an explicit `max_level`, the finest level is chosen for the search radius
        cell_list = TreeCellList(min_corner = (0.0, 0.0), max_corner = (1.0, 2.0))
        nhs = TreeNeighborhoodSearch{2}(; cell_list)
        copy = copy_neighborhood_search(nhs, 0.1, 10)
        @test copy.cell_list.max_level == 0

        # With periodicity.
        periodic_box = PeriodicBox(min_corner = [0.0, 0.0], max_corner = [1.0, 2.0])
        nhs = TreeNeighborhoodSearch{2}(; cell_list, periodic_box)
        copy = copy_neighborhood_search(nhs, 0.1, 10)

        @test copy.periodic_box === periodic_box
        @test copy.cell_list.n_roots == (10, 20)
    end

    @testset "Errors" begin
        cell_list = TreeCellList(min_corner = (0.0, 0.0), max_corner = (1.0, 1.0),
                                 root_cell_size = 0.5, max_level = 3)
        nhs = TreeNeighborhoodSearch{2}(; cell_list, n_points = 2)
        coords = [0.1 0.5; 0.2 0.5]

        error_str = "the number of search radii must match the number of points"
        @test_throws error_str initialize_tree!(nhs, coords, [0.1])

        # Lists are only allocated for `n_points` points
        nhs_small = TreeNeighborhoodSearch{2}(; cell_list, n_points = 1)
        error_str = "the number of points exceeds `n_points` of the neighborhood search"
        @test_throws error_str initialize_tree!(nhs_small, coords, [0.1, 0.1])

        error_str = "search radii must be non-negative"
        @test_throws error_str initialize_tree!(nhs, coords, [0.1, -0.1])

        error_str = "particle coordinates are NaN or outside the domain bounds"
        @test_throws error_str initialize_tree!(nhs, [0.1 1.5; 0.2 0.5], [0.1, 0.1])
        @test_throws error_str initialize_tree!(nhs, [0.1 NaN; 0.2 0.5], [0.1, 0.1])

        # The radii must not exceed the root cell size
        error_str = "the search radii must not exceed the `root_cell_size`"
        @test_throws error_str initialize_tree!(nhs, coords, [0.1, 0.6])

        # Query radii must not exceed the root cell size
        initialize_tree!(nhs, coords, [0.1, 0.1])
        error_str = "must not exceed the root cell size of the tree"
        @test_throws error_str foreach_neighbor((_...) -> nothing, coords, coords,
                                                nhs, 1, search_radius = 0.6)
    end
end;
