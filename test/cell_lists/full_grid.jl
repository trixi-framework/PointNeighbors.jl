@testset "`FullGridCellList`" verbose=true begin
    # Test that `update!` throws an error when a particle is outside the bounding box
    @testset "`update!` bounds check" begin
        @testset "$(N)D" for N in 1:3
            min_corner = fill(0.0, N)
            max_corner = fill(10.0, N)
            search_radius = 1.0

            cell_list = FullGridCellList(; search_radius, min_corner, max_corner)

            # Introduce the same rounding errors for this to pass
            @test cell_list.min_corner == fill(-1.001f0, N)
            @test cell_list.max_corner == fill(10.0 + 1.001f0, N)

            nhs = GridNeighborhoodSearch{N}(; cell_list, search_radius)
            y = rand(N, 10)
            error_nan = ErrorException("particle coordinates contain NaNs")
            error_string_bounds = "particle coordinates are outside the domain bounds of the cell list"
            error_bounds = ErrorException(error_string_bounds)

            y[1, 7] = NaN
            @test_throws error_nan initialize!(nhs, y, y)

            y[1, 7] = -0.01
            @test_throws error_bounds initialize!(nhs, y, y)

            y[1, 7] = 10.01
            @test_throws error_bounds initialize!(nhs, y, y)

            y[1, 7] = 0.0
            @test_nowarn_mod initialize!(nhs, y, y)
            @test_nowarn_mod update!(nhs, y, y)

            y[1, 7] = 10.0
            @test_nowarn_mod update!(nhs, y, y)

            y[1, 7] = NaN
            @test_throws error_nan update!(nhs, y, y)

            y[1, 7] = 10.01
            @test_throws error_bounds update!(nhs, y, y)

            y[1, 7] = -0.01
            @test_throws error_bounds update!(nhs, y, y)
        end
    end
end
