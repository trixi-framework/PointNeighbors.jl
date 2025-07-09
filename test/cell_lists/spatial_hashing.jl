@testset verbose=true "SpatialHashingCellList" begin
    @testset "Collision Handling With Empty Cells" begin
        # The point is in cell (-1, 0) which has a hash collision with cell (-2, -1)
        coordinates = [-0.05; 0.05;;]
        NDIMS = size(coordinates, 1)
        n_points = size(coordinates, 2)
        search_radius = 0.1 + 10 * eps()
        point_index = 1

        nhs = GridNeighborhoodSearch{2}(; search_radius, n_points,
                                        cell_list = SpatialHashingCellList{NDIMS}(n_points))
        initialize_grid!(nhs, coordinates)

        @testset "Test For Collision" begin
            cell1 = (-1, 0)
            cell2 = (-2, -1)
            cell1_hash = PointNeighbors.spatial_hash(cell1, n_points)
            cell2_hash = PointNeighbors.spatial_hash(cell2, n_points)
            points1 = nhs.cell_list[cell1]
            points2 = nhs.cell_list[cell2]

            @test points1 == points2 == [1]
            @test cell1_hash == cell2_hash
        end

        neighbors = Int[]
        foreach_neighbor(coordinates, coordinates, nhs,
                         point_index) do point, neighbor, pos_diff, distance
            push!(neighbors, neighbor)
        end

        @test neighbors == [1]
    end

    @testset "Collision Handling With Non-Empty Cells" begin
        # Cell (-1, 0) with point 1 has a hash collision with cell (-2, -1) with point 2
        coordinates = [[-0.05 -0.15]; [0.05 -0.05]]
        NDIMS = size(coordinates, 1)
        n_points = size(coordinates, 2)
        search_radius = 0.1 + 10 * eps()
        point_index = 1

        nhs = GridNeighborhoodSearch{2}(; search_radius, n_points,
                                        cell_list = SpatialHashingCellList{NDIMS}(n_points))
        initialize_grid!(nhs, coordinates)

        @testset "Test For Collision" begin
            cell1 = (-1, 0)
            cell2 = (-2, -1)
            cell1_hash = PointNeighbors.spatial_hash(cell1, n_points)
            cell2_hash = PointNeighbors.spatial_hash(cell2, n_points)
            points1 = nhs.cell_list[cell1]
            points2 = nhs.cell_list[cell2]

            @test sort(points1) == sort(points2) == [1, 2]
            @test cell1_hash == cell2_hash
        end

        neighbors = [Int[] for _ in axes(coordinates, 2)]
        foreach_point_neighbor(coordinates, coordinates, nhs,
                               points = axes(coordinates, 2)) do point, neighbor, pos_diff,
                                                                 distance
            push!(neighbors[point], neighbor)
        end

        @test neighbors[1] == [1]
        @test neighbors[2] == [2]
    end

    @testset "Cell Coordinates Hash Function" begin
        # 1D coordinates
        @test coordinates_hash([1]) == UInt128(reinterpret(UInt32, Int32(1)))
        @test coordinates_hash([-1]) == UInt128(reinterpret(UInt32, Int32(-1)))
        @test coordinates_hash([0]) == Int128(0)

        # 2D coordinates
        coord2 = [-1, 1]
        hash2 = (UInt128(reinterpret(UInt32, Int32(coord2[2]))) << 32) |
                UInt128(reinterpret(UInt32, Int32(coord2[1])))
        @test coordinates_hash(coord2) == hash2

        # 3D coordinates
        coord3 = [1, 0, -1]
        hash3 = (UInt128(reinterpret(UInt32, Int32(coord3[3]))) << 64) |
                (UInt128(reinterpret(UInt32, Int32(coord3[2]))) << 32) |
                UInt128(reinterpret(UInt32, Int32(coord3[1])))
        @test coordinates_hash(coord3) == hash3

        # Extreme Int32 bounds
        max_val = Int32(typemax(Int32))
        min_val = Int32(typemin(Int32))
        @test coordinates_hash((max_val)) == UInt128(reinterpret(UInt32, max_val))
        @test coordinates_hash((min_val)) == UInt128(reinterpret(UInt32, min_val))

        # 3D extreme Int32 bounds
        coord_ex = [min_val, max_val, Int32(0)]
        hash_ex = (UInt128(reinterpret(UInt32, coord_ex[3])) << (2*32)) |
                  (UInt128(reinterpret(UInt32, coord_ex[2])) << 32) |
                  UInt128(reinterpret(UInt32, coord_ex[1]))
        @test coordinates_hash(coord_ex) == hash_ex

        # Passing non-Int32-coercible should error
        large_val = typemax(Int32) + 1
        @test_throws InexactError coordinates_hash([large_val])

        small_val = typemin(Int32) - 1
        @test_throws InexactError coordinates_hash([small_val])

        @test_throws InexactError coordinates_hash([Inf])
        @test_throws InexactError coordinates_hash([NaN])

        # Too many dimensions should throw assertion
        @test_throws AssertionError coordinates_hash([1, 2, 3, 4])
    end
end
