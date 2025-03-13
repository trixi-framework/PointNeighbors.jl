@testset verbose=true "SpatialHashingCellList" begin
    @testset "Collisions with foreach_neighbor" begin
        range = -0.25:0.1:0.35
        coordinates1 = hcat(collect.(Iterators.product(range, range))...)
        n_points = size(coordinates1, 2)
        search_radius = 0.1

        # point_position1 = [0.05, 0.05]
        # point_index1 = findfirst(row -> row == point_position1, eachcol(coordinates1))
        point_index1 = 25

        nhs1 = GridNeighborhoodSearch{2}(; search_radius, n_points,
                                         cell_list = SpatialHashingCellList{2}(2 * n_points))
        initialize_grid!(nhs1, coordinates1)

        # TODO: test if there is a collision at the point we test, else the test does not make much sense
        # cell = cell_coords(point_position1, nhs1)
        # index = spatial_hash(cell, nhs.cell_list.list_size)
        # @test nhs1.cell_list.cell_collision[index] = true

        function test_neighbor_function(point, neighbor, pos_diff, distance)
            global found_neighbors1
            push!(found_neighbors1, neighbor)
        end

        global found_neighbors1 = Int[]
        foreach_neighbor(test_neighbor_function, coordinates1, coordinates1, nhs1,
                         point_index1)

        correct_neighbors1 = [18, 24, 25, 26, 32]
        @test correct_neighbors1 == found_neighbors1
    end
end;
