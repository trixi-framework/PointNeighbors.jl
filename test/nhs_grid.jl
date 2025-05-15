@testset verbose=true "GridNeighborhoodSearch" begin
    @testset "Constructor" begin
        error_str = "is not a valid update strategy"
        @test_throws "test $error_str" GridNeighborhoodSearch{2}(update_strategy = :test)

        # Default cell list doesn't support fully parallel update
        @test_throws "ParallelUpdate() $error_str" GridNeighborhoodSearch{2}(update_strategy = ParallelUpdate())

        nhs = GridNeighborhoodSearch{3}(update_strategy = SerialUpdate())
        nhs2 = @trixi_test_nowarn PointNeighbors.Adapt.adapt_structure(Array, nhs)

        @test nhs2.update_strategy == nhs.update_strategy
    end

    @testset "`copy_neighborhood_search" begin
        # Basic copy
        nhs = GridNeighborhoodSearch{2}()
        copy = copy_neighborhood_search(nhs, 1.0, 10)

        @test ndims(copy) == 2
        @test PointNeighbors.search_radius(copy) == 1.0
        @test copy.cell_list isa DictionaryCellList{2}
        @test copy.update_strategy == SemiParallelUpdate()

        # Full grid cell list
        min_corner = (0.0, 0.0)
        max_corner = (1.0, 1.0)
        nhs = GridNeighborhoodSearch{2}(cell_list = FullGridCellList(; min_corner,
                                                                     max_corner))
        copy = copy_neighborhood_search(nhs, 1.0, 10)

        @test copy.cell_list isa FullGridCellList
        @test copy.cell_list.cells isa PointNeighbors.DynamicVectorOfVectors
        @test copy.update_strategy == ParallelIncrementalUpdate()

        # Full grid cell list with `Vector{Vector}` backend
        nhs = GridNeighborhoodSearch{2}(cell_list = FullGridCellList(; min_corner,
                                                                     max_corner,
                                                                     backend = Vector{Vector{Int32}}))
        copy = copy_neighborhood_search(nhs, 0.5, 27)

        @test copy.cell_list.cells isa Vector
        @test copy.update_strategy == SemiParallelUpdate()

        # Check that the update strategy is preserved
        nhs = GridNeighborhoodSearch{2}(cell_list = FullGridCellList(; min_corner,
                                                                     max_corner,
                                                                     max_points_per_cell = 101),
                                        update_strategy = SerialUpdate())
        copy = copy_neighborhood_search(nhs, 1.0, 10)

        @test copy.update_strategy == SerialUpdate()
        @test size(copy.cell_list.cells.backend, 1) == 101
    end

    @testset "Cells at Coordinate Limits" begin
        # Test the threshold for very large and very small coordinates
        coords1 = [Inf, -Inf]
        coords2 = [NaN, 0]
        coords3 = [typemax(Int) + 1.0, -typemax(Int) - 1.0]

        @test PointNeighbors.cell_coords(coords1, nothing, nothing, (1.0, 1.0)) ==
              (typemax(Int), typemin(Int))
        @test PointNeighbors.cell_coords(coords2, nothing, nothing, (1.0, 1.0)) ==
              (typemax(Int), 0)
        @test PointNeighbors.cell_coords(coords3, nothing, nothing, (1.0, 1.0)) ==
              (typemax(Int), typemin(Int))

        # The full grid cell list adds one to the coordinates to avoid zero-indexing.
        # This corner case is not relevant, as `typemax` coordinates will always be out of
        # bounds for the finite domain of the full grid cell list.
        cell_list = FullGridCellList(min_corner = (0.0, 0.0), max_corner = (1.0, 1.0),
                                     search_radius = 1.0)

        @test PointNeighbors.cell_coords(coords1, nothing, cell_list, (1.0, 1.0)) ==
              (typemax(Int), typemin(Int)) .+ 1
        @test PointNeighbors.cell_coords(coords2, nothing, cell_list, (1.0, 1.0)) ==
              (typemax(Int), 1) .+ 1
        @test PointNeighbors.cell_coords(coords3, nothing, cell_list, (1.0, 1.0)) ==
              (typemax(Int), typemin(Int)) .+ 1
    end

    @testset "Rectangular Point Cloud 2D" begin
        #### Setup
        # Rectangle of equidistantly spaced points
        # from (x, y) = (-0.25, -0.25) to (x, y) = (0.35, 0.35).
        range = -0.25:0.1:0.35
        coordinates1 = hcat(collect.(Iterators.product(range, range))...)
        n_points = size(coordinates1, 2)

        point_position1 = [0.05, 0.05]
        search_radius = 0.1

        # Create neighborhood search
        nhs1 = GridNeighborhoodSearch{2}(; search_radius, n_points)

        initialize_grid!(nhs1, coordinates1)

        # Get each neighbor for `point_position1`
        neighbors1 = sort(collect(PointNeighbors.eachneighbor(point_position1, nhs1)))

        # Move points
        coordinates2 = coordinates1 .+ [1.4, -3.5]

        # Update neighborhood search
        update_grid!(nhs1, coordinates2)

        # Get each neighbor for updated NHS
        neighbors2 = sort(collect(PointNeighbors.eachneighbor(point_position1, nhs1)))

        # Change position
        point_position2 = point_position1 .+ [1.4, -3.5]

        # Get each neighbor for `point_position2`
        neighbors3 = sort(collect(PointNeighbors.eachneighbor(point_position2, nhs1)))

        # Double search radius
        nhs2 = GridNeighborhoodSearch{2}(search_radius = 2 * search_radius,
                                         n_points = size(coordinates1, 2))
        initialize!(nhs2, coordinates1, coordinates1)

        # Get each neighbor in double search radius
        neighbors4 = sort(collect(PointNeighbors.eachneighbor(point_position1, nhs2)))

        # Move points
        coordinates2 = coordinates1 .+ [0.4, -0.4]

        # Update neighborhood search
        update!(nhs2, coordinates2, coordinates2)

        # Get each neighbor in double search radius
        neighbors5 = sort(collect(PointNeighbors.eachneighbor(point_position1, nhs2)))

        #### Verification against lists of potential neighbors built by hand
        @test neighbors1 == [17, 18, 19, 24, 25, 26, 31, 32, 33]

        @test neighbors2 == Int[]

        @test neighbors3 == [17, 18, 19, 24, 25, 26, 31, 32, 33]

        @test neighbors4 == [
            9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 30, 31,
            32, 33, 34, 35, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49]

        @test neighbors5 == [36, 37, 38, 43, 44, 45]
    end

    @testset verbose=true "Rectangular Point Cloud 3D" begin
        #### Setup
        # Rectangle of equidistantly spaced points
        # from (x, y, z) = (-0.25, -0.25, -0.25) to (x, y, z) = (0.35, 0.35, 0.35).
        range = -0.25:0.1:0.35
        coordinates1 = hcat(collect.(Iterators.product(range, range, range))...)
        n_points = size(coordinates1, 2)

        point_position1 = [0.05, 0.05, 0.05]
        search_radius = 0.1

        # Create neighborhood search
        nhs1 = GridNeighborhoodSearch{3}(; search_radius, n_points)
        initialize_grid!(nhs1, coordinates1)

        # Get each neighbor for `point_position1`
        neighbors1 = sort(collect(PointNeighbors.eachneighbor(point_position1, nhs1)))

        # Move points
        coordinates2 = coordinates1 .+ [1.4, -3.5, 0.8]

        # Update neighborhood search
        update_grid!(nhs1, coordinates2)

        # Get each neighbor for updated NHS
        neighbors2 = sort(collect(PointNeighbors.eachneighbor(point_position1, nhs1)))

        # Change position
        point_position2 = point_position1 .+ [1.4, -3.5, 0.8]

        # Get each neighbor for `point_position2`
        neighbors3 = sort(collect(PointNeighbors.eachneighbor(point_position2, nhs1)))

        #### Verification against lists of potential neighbors built by hand
        @test neighbors1 ==
              [115, 116, 117, 122, 123, 124, 129, 130, 131, 164, 165, 166, 171, 172,
            173, 178, 179, 180, 213, 214, 215, 220, 221, 222, 227, 228, 229]

        @test neighbors2 == Int[]

        @test neighbors3 ==
              [115, 116, 117, 122, 123, 124, 129, 130, 131, 164, 165, 166, 171, 172,
            173, 178, 179, 180, 213, 214, 215, 220, 221, 222, 227, 228, 229]

        update_strategies = (SerialUpdate(), ParallelUpdate())
        @testset verbose=true "eachindex_y $update_strategy" for update_strategy in update_strategies
            # Test that `eachindex_y` is passed correctly to the neighborhood search.
            # This requires `SerialUpdate` or `ParallelUpdate`.
            min_corner = min.(minimum(coordinates1, dims = 2),
                              minimum(coordinates2, dims = 2))
            max_corner = max.(maximum(coordinates1, dims = 2),
                              maximum(coordinates2, dims = 2))
            cell_list = FullGridCellList(; min_corner, max_corner, search_radius)
            nhs2 = GridNeighborhoodSearch{3}(; search_radius, n_points, update_strategy,
                                             cell_list)

            # Initialize with all points
            initialize!(nhs2, coordinates1, coordinates1)

            # Update with a subset of points
            update!(nhs2, coordinates2, coordinates2; eachindex_y = 120:220)

            neighbors2 = sort(collect(PointNeighbors.eachneighbor(point_position1, nhs2)))
            neighbors3 = sort(collect(PointNeighbors.eachneighbor(point_position2, nhs2)))

            # Check that the neighbors are the intersection of the previous neighbors
            # with the `eachindex_y` range.
            @test neighbors2 == Int[]
            @test neighbors3 ==
                  [122, 123, 124, 129, 130, 131, 164, 165, 166, 171, 172, 173,
                178, 179, 180, 213, 214, 215, 220]
        end
    end

    @testset verbose=true "Periodicity" begin
        # These setups are the same as in `test/neighborhood_search.jl`,
        # but instead of testing the actual neighbors with `foreach_point_neighbor`,
        # we only test the potential neighbors (points in neighboring cells) here.

        # Names, coordinates and corresponding periodic boxes for each test
        names = [
            "Simple Example 2D",
            "Box Not Multiple of Search Radius 2D",
            "Simple Example 3D"
        ]

        coordinates = [
            [-0.08 0.0 0.18 0.1 -0.08
             -0.12 -0.05 -0.09 0.15 0.39],
            [-0.08 0.0 0.18 0.1 -0.08
             -0.12 -0.05 -0.09 0.15 0.42],
            [-0.08 0.0 0.18 0.1 -0.08
             -0.12 -0.05 -0.09 0.15 0.39
             0.14 0.34 0.12 0.06 0.13]
        ]

        periodic_boxes = [
            PeriodicBox(min_corner = [-0.1, -0.2], max_corner = [0.2, 0.4]),
            # The `GridNeighborhoodSearch` is forced to round up the cell sizes in this test
            # to avoid split cells.
            PeriodicBox(min_corner = [-0.1, -0.2], max_corner = [0.205, 0.43]),
            PeriodicBox(min_corner = [-0.1, -0.2, 0.05], max_corner = [0.2, 0.4, 0.35])
        ]

        @testset verbose=true "$(names[i])" for i in eachindex(names)
            coords = coordinates[i]

            nhs = GridNeighborhoodSearch{size(coords, 1)}(search_radius = 0.1,
                                                          n_points = size(coords, 2),
                                                          periodic_box = periodic_boxes[i])

            initialize_grid!(nhs, coords)

            neighbors = [sort(collect(PointNeighbors.eachneighbor(coords[:, i], nhs)))
                         for i in 1:5]

            # Note that (1, 2) and (2, 3) are not neighbors, but they are in neighboring cells
            @test neighbors[1] == [1, 2, 3, 5]
            @test neighbors[2] == [1, 2, 3]
            @test neighbors[3] == [1, 2, 3]
            @test neighbors[4] == [4]
            @test neighbors[5] == [1, 5]
        end

        @testset "Offset Domain Triggering Split Cells" begin
            # This test used to trigger a "split cell bug", where the left and right
            # boundary cells were only partially contained in the domain.
            # The left point was placed inside a ghost cells, which caused it to not
            # see the right point, even though it was within the search distance.
            # The domain size is an integer multiple of the cell size, but the NHS did not
            # offset the grid based on the domain position.
            # See https://github.com/trixi-framework/TrixiParticles.jl/pull/211
            # for a more detailed explanation.
            coords = [-1.4 1.9
                      0.0 0.0]

            # 5 x 1 cells
            nhs = GridNeighborhoodSearch{2}(search_radius = 1.0, n_points = size(coords, 2),
                                            periodic_box = PeriodicBox(min_corner = [
                                                                           -1.5,
                                                                           0.0
                                                                       ],
                                                                       max_corner = [
                                                                           2.5,
                                                                           3.0
                                                                       ]))

            initialize_grid!(nhs, coords)

            neighbors = [sort(unique(collect(PointNeighbors.eachneighbor(coords[:, i],
                                                                         nhs))))
                         for i in 1:2]

            @test neighbors[1] == [1, 2]
            @test neighbors[2] == [1, 2]
        end
    end
end;
