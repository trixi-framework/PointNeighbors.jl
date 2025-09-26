@testset verbose=true "SpatialHashingCellList" begin
    @testset "Collision Handling With Empty Cells" begin
        # The point is in cell (-1, 0) which has a hash collision with cell (-2, -1)
        coordinates = [-0.05; 0.05;;]
        NDIMS, n_points = size(coordinates)
        search_radius = 0.1 + 10 * eps()
        point_index = 1

        nhs = GridNeighborhoodSearch{2}(; search_radius, n_points,
                                        cell_list = SpatialHashingCellList{NDIMS}(list_size = n_points))

        @trixi_test_nowarn PointNeighbors.Adapt.adapt_structure(Array, nhs.cell_list)

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
                                        cell_list = SpatialHashingCellList{NDIMS}(list_size = n_points))
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
        @test coordinates_flattened([1]) == UInt128(reinterpret(UInt32, Int32(1)))
        @test coordinates_flattened([-1]) == UInt128(reinterpret(UInt32, Int32(-1)))
        @test coordinates_flattened([0]) == Int128(0)

        # 2D coordinates
        coord2 = [-1, 1]

        # The first coordinate -1 gives the unsigned `UInt32` value 2^32 - 1.
        # The second coordinate gives 1 shifted by 32 bits, so 1 * 2^32.
        expected2 = UInt128(2^32 - 1 + 2^32)

        @test coordinates_flattened(coord2) == expected2

        # 3D coordinates
        coord3 = [1, 0, -1]
        expected3 = UInt128(1 + 0 * 2^32 + (2^32 - 1) * Int128(2)^64)
        @test coordinates_flattened(coord3) == expected3

        # Extreme Int32 bounds
        max_val = Int32(typemax(Int32))
        min_val = Int32(typemin(Int32))
        @test coordinates_flattened((max_val,)) == UInt128(reinterpret(UInt32, max_val))
        @test coordinates_flattened((min_val,)) == UInt128(reinterpret(UInt32, min_val))

        # 3D extreme Int32 bounds
        coord4 = [min_val, max_val, Int32(0)]

        # `typemin(Int32)` gives the unsigned value 2^31.
        # `typemax(Int32)` gives the unsigned value 2^31 - 1.
        expected4 = UInt128(2^31 + (2^31 - 1) * 2^32)

        @test coordinates_flattened(coord4) == expected4

        # Passing non-Int32-coercible should error
        large_val = typemax(Int32) + 1
        @test_throws InexactError coordinates_flattened([large_val])

        small_val = typemin(Int32) - 1
        @test_throws InexactError coordinates_flattened([small_val])

        @test_throws InexactError coordinates_flattened([Inf])
        @test_throws InexactError coordinates_flattened([NaN])

        # Too many dimensions should throw assertion error
        @test_throws AssertionError coordinates_flattened([1, 2, 3, 4])
    end
end
