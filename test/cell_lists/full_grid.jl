@testset "`FullGridCellList`" verbose=true begin
    # Test that an error is thrown when too many dimensions are used
    @testset "constructor" begin
        min_corner = zeros(101)
        max_corner = ones(101)
        search_radius = 1.0

        error_string = "FullGridCellList only supports up to 100 dimensions"
        @test_throws error_string FullGridCellList(; min_corner, max_corner)
        @test_throws error_string FullGridCellList(; min_corner, max_corner, search_radius)

        min_corner = zeros(3)
        max_corner = ones(2)
        error_string = "min_corner and max_corner must have the same length"
        @test_throws error_string FullGridCellList(; min_corner, max_corner)
    end

    # Test that `update!` throws an error when a particle is outside the bounding box
    @testset "`update!` bounds check" begin
        @testset "$(N)D" for N in 1:3
            min_corner = fill(0.0, N)
            max_corner = fill(10.0, N)
            search_radius = 1.0

            cell_list = FullGridCellList(; search_radius, min_corner, max_corner)

            # Introduce the same rounding errors for this to pass
            @test cell_list.min_corner == fill(-1.001, N)
            @test cell_list.max_corner == fill(10.0 + 1.001, N)

            nhs = GridNeighborhoodSearch{N}(; cell_list, search_radius)
            y = rand(N, 10)
            error_string = "particle coordinates are NaN or outside the domain bounds of the cell list"

            y[1, 7] = NaN
            @test_throws error_string initialize!(nhs, y, y)

            y[1, 7] = min_corner[1] - 0.01
            @test_throws error_string initialize!(nhs, y, y)

            # A bit more than max corner might still be inside the grid,
            # but one search radius more is always outside.
            # Also accounting for 0.001 extra padding (see above).
            y[1, 7] = max_corner[1] + 1.01
            @test_throws error_string initialize!(nhs, y, y)

            y[1, 7] = 0.0
            @trixi_test_nowarn initialize!(nhs, y, y)
            @trixi_test_nowarn update!(nhs, y, y)

            y[1, 7] = 10.0
            @trixi_test_nowarn update!(nhs, y, y)

            y[1, 7] = NaN
            @test_throws error_string update!(nhs, y, y)

            # A bit more than max corner might still be inside the grid,
            # but one search radius more is always outside.
            # Also accounting for 0.001 extra padding (see above).
            y[1, 7] = max_corner[1] + 1.01
            @test_throws error_string update!(nhs, y, y)

            y[1, 7] = min_corner[1] - 0.01
            @test_throws error_string update!(nhs, y, y)
        end
    end
end
