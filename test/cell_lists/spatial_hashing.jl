@testset verbose=true "SpatialHashingCellList" begin
    @testset "Collisions with foreach_neighbor" begin
        range = -0.35:0.1:0.25
        coordinates1 = hcat(collect.(Iterators.product(range, range))...)
        n_points = size(coordinates1, 2)
        search_radius = 0.1 + 10 * eps()
        point_index1 = 25
        point_position1 = coordinates1[point_index1]

        @testset verbose=true "List Size $(list_size)" for list_size in [
            1,
            n_points,
            2 * n_points,
            4 * n_points
        ]
            nhs1 = GridNeighborhoodSearch{2}(; search_radius, n_points,
                                             cell_list = SpatialHashingCellList{2}(list_size))
            initialize_grid!(nhs1, coordinates1)

            @testset verbose=true "Collision at test point" begin
                cell = PointNeighbors.cell_coords(point_position1, nhs1)
                index = PointNeighbors.spatial_hash(cell, nhs1.cell_list.list_size)
                @test nhs1.cell_list.cell_collision[index] == true
            end

            found_neighbors1 = Int[]
            foreach_neighbor(coordinates1, coordinates1, nhs1,
                             point_index1) do point, neighbor, pos_diff, distance
                push!(found_neighbors1, neighbor)
            end

            correct_neighbors1 = [18, 24, 25, 26, 32]
            @test correct_neighbors1 == found_neighbors1
        end
    end
end;
