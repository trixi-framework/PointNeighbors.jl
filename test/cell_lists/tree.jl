@testset verbose=true "TreeCellList" begin
    @testset "Initialization with a single level" begin 
        coords = [0.0 1.0 0.3 0.4;
                0.0 0.7 0.6 0.6]
        min_corner = Tuple(minimum(eachcol(coords)))
        max_corner = Tuple(maximum(eachcol(coords)))
        n_dims, n_particles = size(coords)
        cell_list = TreeCellList{n_dims}(; min_corner, max_corner, n_particles, depth=1)
        nhs = TreeNeighborhoodSearch{n_dims}(cell_list)
        PointNeighbors.initialize_tree!(nhs, coords)

        @test cell_list.particle_z == [UInt64(0), UInt64(2), UInt64(2), UInt64(3)]
        @test cell_list.cell_z == [UInt64(0), UInt64(2), UInt64(3)]
        @test cell_list.cell_ranges == [1:1, 2:3, 4:4]
        @test cell_list.cell_levels == [1, 1, 1]
    end
    
    @testset "Initialization with two levels" begin 
        coords = [0.0 1.0 0.30 0.31 0.50;
                  0.0 0.7 0.60 0.60 0.60]
        min_corner = Tuple(minimum(eachcol(coords)))
        max_corner = Tuple(maximum(eachcol(coords)))
        n_dims, n_particles = size(coords)
        cell_list = TreeCellList{n_dims}(; min_corner, max_corner, n_particles, depth=2)
        nhs = TreeNeighborhoodSearch{n_dims}(cell_list)    
        PointNeighbors.initialize_tree!(nhs, coords)

        @test cell_list.particle_z == [UInt64(0), UInt64(9), UInt64(9), UInt64(12), UInt64(13)]
        @test cell_list.cell_z == [UInt64(0), UInt64(9), UInt64(12), UInt64(13)]
        @test cell_list.cell_ranges == [1:1, 2:3, 4:4, 5:5]
        @test cell_list.cell_levels == [1, 2, 2, 2]
    end

    @testset "Neighborhoodsearch" begin
        coords = [0.0 1.0 1.0 0.0 0.7 0.7;
                  0.0 1.0 1.0 0.7 0.4 0.0]
        min_corner = Tuple(minimum(eachcol(coords)))
        max_corner = Tuple(maximum(eachcol(coords)))
        n_dims, n_particles = size(coords)
        cell_list = TreeCellList{n_dims}(; min_corner, max_corner, n_particles, depth=2)
        search_radius = 2 * cell_list.min_cell_length
        nhs = TreeNeighborhoodSearch{n_dims}(cell_list, search_radius)    
        PointNeighbors.initialize_tree!(nhs, coords)
        neighboring_cells = PointNeighbors.neighboring_cells(coords[:,1], nhs)

        @test neighboring_cells == BitSet((1,2,3,4))
    end
end