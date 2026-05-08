@testset verbose=true "TreeCellList" begin
    @testset "`initialize_tree!` with a single level" begin
        coords = [0.0 1.0 0.3 0.4;
                  0.0 1.0 0.6 0.6]
        min_corner = Tuple(minimum(eachcol(coords)))
        max_corner = Tuple(maximum(eachcol(coords)))
        n_dims, n_points = size(coords)
        cell_list = TreeCellList{n_dims}(; min_corner, max_corner, max_level = 1)
        (; cell_levels, marked_cells, active_cells) = cell_list

        search_radius = 2 * cell_list.min_cell_length
        nhs = TreeNeighborhoodSearch{n_dims}(; cell_list, search_radius, n_points)

        PointNeighbors.initialize_tree!(nhs, coords)

        @test active_cells[1] == [1]
        @test active_cells[2] == []
        @test active_cells[3] == [3, 4]
        @test active_cells[4] == [2]
        @test cell_levels == [UInt8(1), UInt8(1), UInt8(1), UInt8(1)]
        @test all(marked_cells .== false)
    end
    @testset "`mark_merge`" begin
        # Z:      1   3    12  19  27  64
        coords = [0.0 0.0  0.1 0.6 0.6 1.0;
                  0.0 0.15 0.4 0.1 0.4 1.0]
        min_corner = Tuple(minimum(eachcol(coords)))
        max_corner = Tuple(maximum(eachcol(coords)))
        n_dims, n_points = size(coords)
        cell_list = TreeCellList{n_dims}(; min_corner, max_corner, max_level = 3)
        (; cell_levels, active_cells) = cell_list

        search_radius = 2 * cell_list.min_cell_length
        nhs = TreeNeighborhoodSearch{n_dims}(; cell_list, search_radius, n_points)

        PointNeighbors.initialize_tree!(nhs, coords)
        PointNeighbors.mark_merge!(cell_list, level=2, capacity=3)
        PointNeighbors.apply_merge!(cell_list, level=2, capacity=3)

        @test active_cells[1] == [1, 2]
        @test active_cells[12] == [3]
        @test active_cells[19] == [4]
        @test active_cells[27] == [5]
        @test active_cells[64] == [6]

        PointNeighbors.mark_merge!(cell_list, level=1, capacity=3)
        PointNeighbors.apply_merge!(cell_list, level=1, capacity=3)

        @test active_cells[1] == [1, 2, 3]
        @test active_cells[19] == [4, 5]
        @test active_cells[64] == [6]      # This point should not get merged
    end

    @testset "`mark_refine`" begin
        # Z:      1   3    12  19  27  64
        coords = [0.0 0.0  0.1 0.6 0.6 1.0;
                  0.0 0.15 0.4 0.1 0.4 1.0]
        min_corner = Tuple(minimum(eachcol(coords)))
        max_corner = Tuple(maximum(eachcol(coords)))
        n_dims, n_points = size(coords)
        cell_list = TreeCellList{n_dims}(; min_corner, max_corner, max_level = 3)
        (; cell_levels, active_cells) = cell_list

        search_radius = 2 * cell_list.min_cell_length
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

        PointNeighbors.mark_refine!(cell_list, level=1)
        PointNeighbors.apply_refine!(cell_list, nhs, coords, level=1)

        @test active_cells[1] == [1,2]
        @test cell_levels[1] == 2
        @test active_cells[9] == [3]
        @test cell_levels[9] == 2
        @test active_cells[17] == [4]
        @test cell_levels[17] == 2
        @test active_cells[25] == [5]
        @test cell_levels[25] == 2
        @test active_cells[61] == [6]
        @test cell_levels[61] == 2

        PointNeighbors.mark_refine!(cell_list, level=2)
        PointNeighbors.apply_refine!(cell_list, nhs, coords, level=2)

        @test active_cells[1] == [1]
        @test cell_levels[1] == 3
        @test active_cells[3] == [2]
        @test cell_levels[1] == 3
    end

    @testset "`initialize_tree!` with two levels" begin
        coords = [0.0 1.0 0.30 0.31 0.50;
                  0.0 0.7 0.60 0.60 0.60]
        min_corner = Tuple(minimum(eachcol(coords)))
        max_corner = Tuple(maximum(eachcol(coords)))
        n_dims, n_particles = size(coords)
        cell_list = TreeCellList{n_dims}(; min_corner, max_corner, n_particles, level = 2)
        nhs = TreeNeighborhoodSearch{n_dims}(cell_list)
        PointNeighbors.initialize_tree!(nhs, coords)

        @test cell_list.particle_z ==
              [UInt64(0), UInt64(9), UInt64(9), UInt64(12), UInt64(13)]
        @test cell_list.cell_z == [UInt64(0), UInt64(9), UInt64(12), UInt64(13)]
        @test cell_list.cell_ranges == [1:1, 2:3, 4:4, 5:5]
        @test cell_list.cell_levels == [1, 2, 2, 2]
    end

    @testset "`neighboring_cells`" begin
        coords = [0.0 1.0 1.0 0.0 0.7 0.7;
                  0.0 1.0 1.0 0.7 0.4 0.0]
        min_corner = Tuple(minimum(eachcol(coords)))
        max_corner = Tuple(maximum(eachcol(coords)))
        n_dims, n_particles = size(coords)
        cell_list = TreeCellList{n_dims}(; min_corner, max_corner, n_particles, level = 2)
        search_radius = 2 * cell_list.min_cell_length
        nhs = TreeNeighborhoodSearch{n_dims}(cell_list, search_radius)
        PointNeighbors.initialize_tree!(nhs, coords)
        neighboring_cells = PointNeighbors.neighboring_cells(coords[:, 1], nhs)

        @test neighboring_cells == BitSet((1, 2, 3, 4))
    end

    @testset "`foreach_neighbor`" begin
        coords = [0.0 1.0 1.0 0.0 0.7 0.7;
                  0.0 1.0 1.0 0.7 0.4 0.0]
        min_corner = Tuple(minimum(eachcol(coords)))
        max_corner = Tuple(maximum(eachcol(coords)))
        n_dims, n_particles = size(coords)
        cell_list = TreeCellList{n_dims}(; min_corner, max_corner, n_particles, level = 2)
        search_radius = 2 * cell_list.min_cell_length
        nhs = TreeNeighborhoodSearch{n_dims}(cell_list, search_radius)
        PointNeighbors.initialize_tree!(nhs, coords)

        neighbors = Vector{Int}(undef, 0)
        point = 1
        point_coords = coords[:, particle]
        PointNeighbors.foreach_neighbor(coords, nhs, point, point_coords,
                                        nhs.search_radius) do particle, neighbor, pos_diff,
                                                              distance
            push!(neighbors, neighbor)
        end

        @test sort!(neighbors) == [1, 4, 6]
    end

    @testset "`initialize!`" begin
        coords = [0.0 1.0 1.0 0.0 0.7 0.7;
                  0.0 1.0 1.0 0.7 0.4 0.0]
        min_corner = Tuple(minimum(eachcol(coords)))
        max_corner = Tuple(maximum(eachcol(coords)))
        n_dims, n_particles = size(coords)
        cell_list = TreeCellList{n_dims}(; min_corner, max_corner, n_particles, level = 2)
        search_radius = 2 * cell_list.min_cell_length
        nhs = TreeNeighborhoodSearch{n_dims}(cell_list, search_radius)
        initialize!(nhs, coords, coords)

        @test cell_list.particle_z == [UInt64(0), UInt64(2), UInt64(2), UInt64(3)]
        @test cell_list.cell_z == [UInt64(0), UInt64(2), UInt64(3)]
        @test cell_list.cell_ranges == [1:1, 2:3, 4:4]
        @test cell_list.cell_levels == [1, 1, 1]
    end

    @testset "`update!`" begin
        coords = [0.0 1.0 1.0 0.0 0.7 0.7;
                  0.0 1.0 1.0 0.7 0.4 0.0]
        min_corner = Tuple(minimum(eachcol(coords)))
        max_corner = Tuple(maximum(eachcol(coords)))
        n_dims, n_particles = size(coords)
        cell_list = TreeCellList{n_dims}(; min_corner, max_corner, n_particles, level = 2)
        search_radius = 2 * cell_list.min_cell_length
        nhs = TreeNeighborhoodSearch{n_dims}(cell_list, search_radius)
        initialize!(nhs, coords, coords)

        coords = coords .+ [-0.1, 1.0]

        update!(nhs, coords, coords)
    end
end
