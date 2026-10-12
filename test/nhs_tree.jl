@testset verbose=true "TreeNeighborhoodSearch" begin
    # Fixed seed to ensure reproducibility of the random point clouds below
    Random.seed!(1)

    # Brute-force reference: all points `j` in `y` with `|x - y_j| <= h`
    function brute_force_neighbors(x, h, y; eachindex_y = axes(y, 2))
        return [j for j in eachindex_y if sqrt(sum(abs2, x - y[:, j])) <= h]
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

        error_str = "the search radius must not exceed the `root_cell_size`"
        @test_throws error_str TreeNeighborhoodSearch{2}(; cell_list, search_radius = 1.1)

        error_str = "`max_level` must be between 0 and 20 in 3D"
        @test_throws error_str TreeCellList(min_corner = (0.0, 0.0, 0.0),
                                            max_corner = (1.0, 1.0, 1.0), max_level = 21)

        # The element type is determined by the domain corners
        cell_list = TreeCellList(min_corner = (0.0f0, 0.0f0), max_corner = (1.0f0, 2.0f0),
                                 max_level = 3)
        nhs = TreeNeighborhoodSearch{2}(; cell_list, search_radius = 0.1)
        @test eltype(nhs) == Float32
        @test nhs.search_radius isa Float32

        # A single root cell covering the domain by default
        @test cell_list.root_cell_size == 2.0f0
        @test cell_list.n_roots == (1, 1)

        # A grid of 4 x 8 root cells
        cell_list = TreeCellList(min_corner = (0.0, 0.0), max_corner = (1.0, 2.0),
                                 max_level = 3, root_cell_size = 0.25)
        @test cell_list.n_roots == (4, 8)
    end

    @testset "Leaf Level" begin
        # A single root cell of size 1
        cell_list = TreeCellList(min_corner = (0.0, 0.0), max_corner = (1.0, 1.0),
                                 max_level = 3)
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

        # At most at `max_level`
        @test leaf_level(0.01) == 3

        # One list is allocated for each leaf
        nhs = TreeNeighborhoodSearch{2}(; cell_list, search_radius = 0.3)
        @test length(nhs.cell_list.cells) == 4
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

    @testset "Points Stored in Leaves" begin
        coords = rand(2, 500)

        # With a search radius of 0.1, the leaves are at level 3 with a size of 0.125
        cell_list_ = TreeCellList(min_corner = (0.0, 0.0), max_corner = (1.0, 1.0),
                                  max_level = 6)
        nhs = TreeNeighborhoodSearch{2}(; cell_list = cell_list_, search_radius = 0.1)

        # Leave out some points
        eachindex_y = 1:2:500
        initialize!(nhs, coords, coords; eachindex_y)

        # The neighborhood search allocates the lists of points in the leaves
        (; cell_list) = nhs
        @test cell_list.leaf_level == 3
        @test length(cell_list.cells) == 4^3

        points_found = Int[]
        for leaf in 0:(4 ^ 3 - 1)
            points = PointNeighbors.points_in_leaf(cell_list, leaf)
            for point in points
                # The point must be inside this leaf
                leaf_cell = floor.(Int, coords[:, point] ./ 0.125)
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
        # on every level
        coords[:, 1] .= 0.5

        # Different tree configurations
        min_corner = zeros(NDIMS)
        max_corner = ones(NDIMS)
        cell_lists = [
            "Single Root Cell" => TreeCellList(; min_corner, max_corner, max_level = 6,
                                               max_points_per_cell = n_points),
            "Vector{Vector} Backend" => TreeCellList(; min_corner, max_corner,
                                                     max_level = 6,
                                                     backend = Vector{Vector{Int32}}),
            # Multiple root cells, which must be at least as large as the search radius
            "Root Grid" => TreeCellList(; min_corner, max_corner, max_level = 3,
                                        root_cell_size = 0.25,
                                        max_points_per_cell = n_points),
            # Leaves that are much larger than the search radius
            "Coarse Leaves" => TreeCellList(; min_corner, max_corner, max_level = 1,
                                            max_points_per_cell = n_points),
            # A domain that is larger than the point cloud
            "Large Domain" => TreeCellList(; min_corner = fill(-1.0, NDIMS),
                                           max_corner = fill(2.0, NDIMS), max_level = 6,
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

    @testset "Exact Distances on a Lattice $(NDIMS)D" for NDIMS in 2:3
        # Points on a lattice with spacing 1/16, which is exactly representable.
        # Many points are exactly on cell boundaries, and many pairs have a distance of
        # exactly the search radius, which must be found (`distance <= radius`).
        lattice = CartesianIndices(ntuple(_ -> 17, NDIMS))
        coords = reduce(hcat, [collect(Tuple(cell) .- 1) ./ 16 for cell in vec(lattice)])
        search_radius = 2 / 16

        cell_list = TreeCellList(; min_corner = zeros(NDIMS), max_corner = ones(NDIMS),
                                 max_level = 5, max_points_per_cell = size(coords, 2))
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
                                 max_corner = (1.0, 1.0, 1.0), max_level = 5)
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
                                 max_level = 4, root_cell_size = 0.5)
        nhs = TreeNeighborhoodSearch{2}(; cell_list, update_strategy = SerialUpdate())

        copy = copy_neighborhood_search(nhs, 0.1, 10)

        @test copy.search_radius == 0.1
        @test copy.cell_list.min_corner == cell_list.min_corner
        @test copy.cell_list.max_corner == cell_list.max_corner
        @test copy.cell_list.max_level == 4
        @test copy.cell_list.root_cell_size == 0.5
        @test copy.cell_list.cells isa PointNeighbors.DynamicVectorOfVectors
        @test copy.update_strategy == SerialUpdate()

        # The leaves are chosen for the new search radius
        @test nhs.cell_list.leaf_level == 4
        @test copy.cell_list.leaf_level == 2

        # The copy has its own lists
        @test copy.cell_list.cells !== nhs.cell_list.cells
    end

    @testset "Errors" begin
        cell_list = TreeCellList(min_corner = (0.0, 0.0), max_corner = (1.0, 1.0),
                                 max_level = 3)
        nhs = TreeNeighborhoodSearch{2}(; cell_list, search_radius = 0.1)

        error_str = "particle coordinates are NaN or outside the domain bounds"
        @test_throws error_str initialize!(nhs, [0.1 1.5; 0.2 0.5], [0.1 1.5; 0.2 0.5])
        @test_throws error_str initialize!(nhs, [0.1 NaN; 0.2 0.5], [0.1 NaN; 0.2 0.5])
    end
end;
