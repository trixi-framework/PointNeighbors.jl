@testset verbose=true "TreeNeighborhoodSearch" begin
    # Fixed seed to ensure reproducibility of the random point clouds below
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
                                 root_cell_size = 1.0)

        error_str = "a 3D cell list is required for a TreeNeighborhoodSearch{3}"
        @test_throws error_str TreeNeighborhoodSearch{3}(; cell_list)

        error_str = "`TreeNeighborhoodSearch` requires a `TreeCellList`"
        @test_throws error_str TreeNeighborhoodSearch{2}(cell_list = DictionaryCellList{2}())

        error_str = "is not a valid update strategy"
        @test_throws error_str TreeNeighborhoodSearch{2}(; cell_list,
                                                         update_strategy = SemiParallelUpdate())

        error_str = "the search radius must not exceed the `root_cell_size`"
        @test_throws error_str TreeNeighborhoodSearch{2}(; cell_list, search_radius = 1.1)

        # The root cell size is required unless creating a template
        template_cell_list = TreeCellList(min_corner = (0.0, 0.0), max_corner = (1.0, 1.0))
        error_str = "the `root_cell_size` of the `TreeCellList` must be set"
        @test_throws error_str TreeNeighborhoodSearch{2}(cell_list = template_cell_list,
                                                         search_radius = 0.1)

        # The element type is determined by the domain corners
        cell_list = TreeCellList(min_corner = (0.0f0, 0.0f0), max_corner = (1.0f0, 2.0f0),
                                 root_cell_size = 0.25f0)
        nhs = TreeNeighborhoodSearch{2}(; cell_list, search_radius = 0.1)
        @test eltype(nhs) == Float32
        @test nhs.search_radius isa Float32

        # A grid of 4 x 8 root cells covering the domain,
        # with one layer of root cells around it
        @test nhs.cell_list.n_roots == (6, 10)
        @test nhs.cell_list.grid_min_corner == [-0.25f0, -0.25f0]
    end

    @testset "Leaf Level" begin
        cell_list = TreeCellList(min_corner = (0.0, 0.0), max_corner = (1.0, 1.0),
                                 root_cell_size = 1.0)
        function leaf_level(search_radius)
            nhs = TreeNeighborhoodSearch{2}(; cell_list, search_radius)
            return nhs.cell_list.leaf_level
        end

        # The leaves are at the finest level with cells that are at least as large
        # as the search radius
        @test leaf_level(1.0) == 0
        @test leaf_level(0.3) == 1
        @test leaf_level(0.25) == 2
        @test leaf_level(0.2) == 2
        @test leaf_level(0.01) == 6

        # One list is allocated for each leaf of the 3 x 3 root cells
        nhs = TreeNeighborhoodSearch{2}(; cell_list, search_radius = 0.3)
        @test length(nhs.cell_list.cells) == 9 * 4
    end

    @testset "Morton Order" begin
        # A 2D grid of 3 x 2 root cells
        cell_list = TreeCellList(min_corner = (0.0, 0.0), max_corner = (3.0, 2.0),
                                 root_cell_size = 1.0)

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

    @testset "Neighbor Index Parts $(NDIMS)D" for NDIMS in 1:3
        # The parts of the neighboring leaves computed with Morton increments and decrements
        # must match the parts computed with `cell_index_part`, also when the neighboring
        # leaf is in the neighboring root cell
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
        # the leaves at the opposite boundary
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

    @testset "Points Stored in Leaves" begin
        coords = rand(2, 500)

        # With a search radius of 0.1, the leaves are at level 3 with a size of 0.125
        cell_list_ = TreeCellList(min_corner = (0.0, 0.0), max_corner = (1.0, 1.0),
                                  root_cell_size = 1.0)
        nhs = TreeNeighborhoodSearch{2}(; cell_list = cell_list_, search_radius = 0.1)

        # Leave out some points
        eachindex_y = 1:2:500
        initialize!(nhs, coords, coords; eachindex_y)

        # The neighborhood search allocates the lists of points in the leaves
        # of the 3 x 3 root cells
        (; cell_list) = nhs
        @test cell_list.leaf_level == 3
        @test length(cell_list.cells) == 9 * 4^3

        points_found = Int[]
        for leaf in 0:(9 * 4 ^ 3 - 1)
            points = PointNeighbors.points_in_leaf(cell_list, leaf)
            for point in points
                # The point must be inside this leaf. The grid of root cells starts
                # one root cell before the domain.
                leaf_cell = floor.(Int, (coords[:, point] .+ 1) ./ 0.125)
                @test PointNeighbors.cell_index(cell_list, Tuple(leaf_cell), 3) == leaf
            end
            append!(points_found, points)
        end

        # Each active point is stored in exactly one leaf
        @test sort(points_found) == eachindex_y
    end

    @testset verbose=true "Compare Against Brute Force $(NDIMS)D" for NDIMS in 1:3
        n_points = (200, 2_000, 4_000)[NDIMS]
        coords = rand(NDIMS, n_points)

        # Move the first point to the center of the unit cube, which is on a cell boundary
        # on every level, and the second point to the corner of the domain
        coords[:, 1] .= 0.5
        coords[:, 2] .= 1.0

        # Different tree configurations
        min_corner = zeros(NDIMS)
        max_corner = ones(NDIMS)
        cell_lists = [
            "Root Cells of Size 0.125" => TreeCellList(; min_corner, max_corner,
                                                       root_cell_size = 0.125,
                                                       max_points_per_cell = n_points),
            "Vector{Vector} Backend" => TreeCellList(; min_corner, max_corner,
                                                     root_cell_size = 0.125,
                                                     backend = Vector{Vector{Int32}}),
            # Leaves inside the root cells, so that the 3^d block of leaves around a leaf
            # can be in multiple root cells
            "Root Cells of Size 1" => TreeCellList(; min_corner, max_corner,
                                                   root_cell_size = 1.0,
                                                   max_points_per_cell = n_points),
            "Root Cells Not Aligned With Domain" => TreeCellList(; min_corner,
                                                                 max_corner,
                                                                 root_cell_size = 0.3,
                                                                 max_points_per_cell = n_points),
            # Leaves that are larger than the search radius
            "Large Leaves" => TreeCellList(; min_corner, max_corner,
                                           root_cell_size = 0.2,
                                           max_points_per_cell = n_points),
            # A domain that is larger than the point cloud
            "Large Domain" => TreeCellList(; min_corner = fill(-1.0, NDIMS),
                                           max_corner = fill(2.0, NDIMS),
                                           root_cell_size = 0.5,
                                           max_points_per_cell = n_points)
        ]

        # A search radius that is a power of two times the size of the root cells,
        # such that leaves are exactly as large as the search radius,
        # and one that is not
        @testset "$cell_list_name, Search Radius $search_radius" for (cell_list_name,
                                                                      cell_list) in
                                                                     cell_lists,
                                                                     search_radius in
                                                                     (0.125, 0.1)

            nhs = TreeNeighborhoodSearch{NDIMS}(; cell_list, search_radius)

            # Initialize with different coordinates first to also test the update
            initialize!(nhs, coords, rand(NDIMS, n_points))
            update!(nhs, coords, coords)

            # Query each point of the tree
            @test all(axes(coords, 2)) do i
                tree_neighbors(nhs, coords[:, i], search_radius, coords) ==
                brute_force_neighbors(coords[:, i], search_radius, coords)
            end

            # Query arbitrary points, including points outside of the domain,
            # with query radii that are not larger than the search radius
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
        n_points = (200, 2_000, 4_000)[NDIMS]

        # A periodic box that is not a multiple of the root cell size in all dimensions
        min_corner = fill(-0.1, NDIMS)
        max_corner = [0.9, 0.6, 0.85][1:NDIMS]
        periodic_box = PeriodicBox(; min_corner, max_corner)
        coords = min_corner .+ periodic_box.size .* rand(NDIMS, n_points)

        # Points on the boundary of the periodic box
        coords[:, 1] .= min_corner
        coords[:, 2] .= max_corner

        cell_lists = [
            "Root Cells of Size 0.1" => TreeCellList(; min_corner, max_corner,
                                                     root_cell_size = 0.1,
                                                     max_points_per_cell = n_points),
            "Root Cells of Size 0.3" => TreeCellList(; min_corner, max_corner,
                                                     root_cell_size = 0.3,
                                                     max_points_per_cell = n_points),
            # A single root cell in each dimension, so that the neighboring leaves
            # across the periodic boundary are in the same root cell
            "Root Cells of Size 0.6" => TreeCellList(; min_corner, max_corner,
                                                     root_cell_size = 0.6,
                                                     max_points_per_cell = n_points)
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

            # Query arbitrary points, including points outside of the periodic box
            query_coords = min_corner .- 0.5 .+ (periodic_box.size .+ 1) .* rand(NDIMS, 200)
            @test all(axes(query_coords, 2)) do i
                x = query_coords[:, i]
                tree_neighbors(nhs, x, search_radius, coords) ==
                brute_force_neighbors(x, search_radius, coords; periodic_box)
            end
        end
    end

    @testset "Exact Distances on a Lattice $(NDIMS)D" for NDIMS in 2:3
        # Points on a lattice with spacing 1/16, which is exactly representable.
        # Many points are exactly on cell boundaries, and many pairs have a distance of
        # exactly the search radius, which must be found (`distance <= radius`).
        lattice = CartesianIndices(ntuple(_ -> 17, NDIMS))
        coords = reduce(hcat, [collect(Tuple(cell) .- 1) ./ 16 for cell in vec(lattice)])
        search_radius = 2 / 16

        cell_list = TreeCellList(; min_corner = zeros(NDIMS), max_corner = ones(NDIMS),
                                 root_cell_size = 0.5,
                                 max_points_per_cell = size(coords, 2))
        nhs = TreeNeighborhoodSearch{NDIMS}(; cell_list, search_radius)
        initialize!(nhs, coords, coords)

        @test all(axes(coords, 2)) do i
            tree_neighbors(nhs, coords[:, i], search_radius, coords) ==
            brute_force_neighbors(coords[:, i], search_radius, coords)
        end
    end

    @testset "Inactive Points" begin
        coords = rand(3, 1000)

        cell_list = TreeCellList(min_corner = (0.0, 0.0, 0.0),
                                 max_corner = (1.0, 1.0, 1.0), root_cell_size = 0.5)
        nhs = TreeNeighborhoodSearch{3}(; cell_list, search_radius = 0.1)

        # Only every third point is part of the tree
        eachindex_y = 1:3:1000
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

        # The copy has its own lists
        @test copy.cell_list.cells !== nhs.cell_list.cells

        # With periodicity
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

        # The domain must be the periodic box
        periodic_box = PeriodicBox(min_corner = [0.0, 0.0], max_corner = [2.0, 1.0])
        error_str = "the domain of the `TreeCellList` must be the periodic box"
        @test_throws error_str TreeNeighborhoodSearch{2}(; cell_list, periodic_box,
                                                         search_radius = 0.1)

        # At least 3 leaves in each dimension with periodicity
        periodic_box = PeriodicBox(min_corner = [0.0, 0.0], max_corner = [1.0, 1.0])
        error_str = "needs at least 3 leaves in each dimension when used with periodicity"
        @test_throws error_str TreeNeighborhoodSearch{2}(; cell_list, periodic_box,
                                                         search_radius = 0.4)
    end
end;
