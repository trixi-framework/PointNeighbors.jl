@testset verbose=true "TreeNeighborhoodSearch" begin
    @testset "`initialize_tree!`" begin
        # Z:      1   3    12  19  27  64
        coords = [0.0 0.0 0.1 0.6 0.6 1.0;
                  0.0 0.15 0.4 0.1 0.4 1.0]
        min_corner = Tuple(minimum(eachcol(coords)))
        max_corner = Tuple(maximum(eachcol(coords)))
        n_dims, n_points = size(coords)
        cell_list = TreeGridCellList(; min_corner, max_corner, max_level = 3,
                                     capacity_per_cell = 1)

        (; cell_levels, cells, max_level) = cell_list
        length_grid = PointNeighbors.grid_length(cell_list)
        cell_length = length_grid / (2^max_level)
        search_radius = 2 * cell_length

        nhs = TreeNeighborhoodSearch{n_dims}(; cell_list, search_radius, n_points)

        PointNeighbors.initialize_tree!(nhs, coords, iters = 0)

        # Points are placed in their max_level Morton leaf cells
        @test cells[1] == [1]
        @test cells[3] == [2]
        @test cells[12] == [3]
        @test cells[19] == [4]
        @test cells[27] == [5]
        @test cells[64] == [6]

        @test all(cell_levels[[1, 3, 12, 19, 27, 64]] .== 3)

        @test all(nhs.marked_cells .== false)
    end
    @testset "`merge`" begin
        # Z:      1   3    12  19  27  64
        coords = [0.0 0.0 0.1 0.6 0.6 1.0;
                  0.0 0.15 0.4 0.1 0.4 1.0]
        min_corner = Tuple(minimum(eachcol(coords)))
        max_corner = Tuple(maximum(eachcol(coords)))
        n_dims, n_points = size(coords)
        cell_list = TreeGridCellList(; min_corner, max_corner, max_level = 3,
                                     capacity_per_cell = 1)

        (; cell_levels, cells, max_level) = cell_list
        length_grid = PointNeighbors.grid_length(cell_list)
        cell_length = length_grid / (2^max_level)
        search_radius = 2 * cell_length

        nhs = TreeNeighborhoodSearch{n_dims}(; cell_list, search_radius, n_points)

        PointNeighbors.initialize_tree!(nhs, coords, iters = 0)
        PointNeighbors.mark_merge!(nhs, level = 2, capacity = 2)
        PointNeighbors.apply_merge!(nhs, level = 2)

        @test cells[1] == [1, 2]
        @test cells[9] == [3]
        @test cells[17] == [4]
        @test cells[25] == [5]
        @test cells[61] == [6]
        @test all(cell_levels[[1, 9, 17, 25, 61]] .== 2)

        PointNeighbors.mark_merge!(nhs, level = 1, capacity = 2)
        PointNeighbors.apply_merge!(nhs, level = 1)

        @test cells[1] == [1, 2] # Should not get merged
        @test cells[9] == [3]
        @test cells[17] == [4, 5]
        @test cells[49] == [6]

        @test all(cell_levels[[17, 49]] .== 1)
        @test cell_levels[1] == 2
        @test cell_levels[9] == 2
    end

    @testset "`refine`" begin
        # Z:      1   3    12  19  27  64
        coords = [0.0 0.0 0.1 0.6 0.6 1.0;
                  0.0 0.15 0.4 0.1 0.4 1.0]
        min_corner = Tuple(minimum(eachcol(coords)))
        max_corner = Tuple(maximum(eachcol(coords)))
        n_dims, n_points = size(coords)
        cell_list = TreeGridCellList(; min_corner, max_corner, max_level = 3)

        (; cell_levels, cells, max_level) = cell_list
        length_grid = PointNeighbors.grid_length(cell_list)
        cell_length = length_grid / (2^max_level)
        search_radius = 2 * cell_length

        nhs = TreeNeighborhoodSearch{n_dims}(; cell_list, search_radius, n_points)

        PointNeighbors.push_cell!(cell_list, 1, 1)
        PointNeighbors.push_cell!(cell_list, 1, 2)
        PointNeighbors.push_cell!(cell_list, 1, 3)
        PointNeighbors.push_cell!(cell_list, 17, 4)
        PointNeighbors.push_cell!(cell_list, 17, 5)
        PointNeighbors.push_cell!(cell_list, 61, 6)

        cell_levels .= 0
        cell_levels[1] = 1
        cell_levels[17] = 1
        cell_levels[61] = 2

        PointNeighbors.mark_refine!(nhs)
        PointNeighbors.apply_refine!(nhs, coords)

        @test cells[1] == [1, 2]
        @test cell_levels[1] == 2
        @test cells[9] == [3]
        @test cell_levels[9] == 2
        @test cells[17] == [4]
        @test cell_levels[17] == 2
        @test cells[25] == [5]
        @test cell_levels[25] == 2
        @test cells[61] == [6]
        @test cell_levels[61] == 2

        PointNeighbors.mark_refine!(nhs, level = 2)
        PointNeighbors.apply_refine!(nhs, coords, level = 2)

        @test cells[1] == [1]
        @test cell_levels[1] == 3
        @test cells[3] == [2]
        @test cell_levels[1] == 3
    end

    @testset "`foreach_neighbor`" begin
        # Z:      1   3    12  19  27  64
        coords = [0.0 0.0 0.1 0.6 0.6 1.0;
                  0.0 0.15 0.4 0.1 0.4 1.0]
        min_corner = Tuple(minimum(eachcol(coords)))
        max_corner = Tuple(maximum(eachcol(coords)))
        n_dims, n_points = size(coords)
        cell_list = TreeGridCellList(; min_corner, max_corner, max_level = 3,
                                     capacity_per_cell = 1)

        (; cell_levels, cells, max_level) = cell_list
        length_grid = PointNeighbors.grid_length(cell_list)
        cell_length = length_grid / (2^max_level)
        search_radius = 2 * cell_length

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

        @test sort!(neighbors) == [1, 2]
    end
    @testset "`update!`" begin end

    @testset "`foreach_neighbor`" begin
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
            # Z:      1   3    12  19  27  64
            coords = [0.0 0.0 0.1 0.6 0.6 1.0;
                      0.0 0.15 0.4 0.1 0.4 1.0]
            min_corner = Tuple(minimum(eachcol(coords)))
            max_corner = Tuple(maximum(eachcol(coords)))
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

            # Explicit Boundary Check: Point 6 is at the top-right corner (1.0, 1.0)
            @test collect_neighbors(coords, nhs_tree, 6) ==
                  collect_neighbors(coords, nhs_trivial, 6)
        end

        @testset "2D with Merge (`iters > 0`)" begin
            coords = [0.0 0.0 0.1 0.6 0.6 1.0;
                      0.0 0.15 0.4 0.1 0.4 1.0]
            min_corner = Tuple(minimum(eachcol(coords)))
            max_corner = Tuple(maximum(eachcol(coords)))
            n_dims, n_points = size(coords)

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
            coords = [0.0 0.0 0.1 0.6 0.6 1.0;
                      0.0 0.15 0.4 0.1 0.4 1.0]
            min_corner = Tuple(minimum(eachcol(coords)))
            max_corner = Tuple(maximum(eachcol(coords)))
            n_dims, n_points = size(coords)

            cell_list = TreeGridCellList(; min_corner, max_corner, max_level = 3,
                                         capacity_per_cell = 1)
            length_grid = PointNeighbors.grid_length(cell_list)
            cell_length = length_grid / (2^cell_list.max_level)

            # Test small search radius
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

            # Test large search radius
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
            coords_3d = [0.0 0.1 0.5 0.9 1.0;
                         0.0 0.1 0.5 0.9 1.0;
                         0.0 0.1 0.5 0.9 1.0]
            min_corner = Tuple(minimum(eachcol(coords_3d)))
            max_corner = Tuple(maximum(eachcol(coords_3d)))
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
end
