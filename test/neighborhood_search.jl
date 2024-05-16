# This file contains tests for the generic functions in `src/neighborhood_search.jl` and
# tests comparing all NHS implementations against the `TrivialNeighborhoodSearch`.
@testset verbose=true "Neighborhood Searches" begin
    @testset verbose=true "Periodicity" begin
        # These examples are constructed by hand and are therefore a good test for the
        # trivial neighborhood search as well.
        # (As opposed to the tests below that are just comparing against the trivial NHS.)

        # Names, coordinates and corresponding periodic boxes for each test
        names = [
            "Simple Example 2D",
            "Box Not Multiple of Search Radius 2D",
            "Simple Example 3D",
        ]

        coordinates = [
            [-0.08 0.0 0.18 0.1 -0.08
             -0.12 -0.05 -0.09 0.15 0.39],
            [-0.08 0.0 0.18 0.1 -0.08
             -0.12 -0.05 -0.09 0.15 0.42],
            [-0.08 0.0 0.18 0.1 -0.08
             -0.12 -0.05 -0.09 0.15 0.39
             0.14 0.34 0.12 0.06 0.13],
        ]

        periodic_boxes = [
            ([-0.1, -0.2], [0.2, 0.4]),
            # The `GridNeighborhoodSearch` is forced to round up the cell sizes in this test
            # to avoid split cells.
            ([-0.1, -0.2], [0.205, 0.43]),
            ([-0.1, -0.2, 0.05], [0.2, 0.4, 0.35]),
        ]

        @testset verbose=true "$(names[i])" for i in eachindex(names)
            coords = coordinates[i]

            NDIMS = size(coords, 1)
            n_particles = size(coords, 2)
            search_radius = 0.1

            neighborhood_searches = [
                TrivialNeighborhoodSearch{NDIMS}(search_radius, 1:n_particles,
                                                 periodic_box_min_corner = periodic_boxes[i][1],
                                                 periodic_box_max_corner = periodic_boxes[i][2]),
                GridNeighborhoodSearch{NDIMS}(search_radius, n_particles,
                                              periodic_box_min_corner = periodic_boxes[i][1],
                                              periodic_box_max_corner = periodic_boxes[i][2]),
            ]
            neighborhood_searches_names = [
                "`TrivialNeighborhoodSearch`",
                "`GridNeighborhoodSearch`",
            ]

            # Run this for every neighborhood search
            @testset "$(neighborhood_searches_names[j])" for j in eachindex(neighborhood_searches_names)
                nhs = neighborhood_searches[j]

                initialize!(nhs, coords, coords)

                neighbors = [Int[] for _ in axes(coords, 2)]

                for_particle_neighbor(coords, coords, nhs,
                                      particles = axes(coords, 2)) do particle, neighbor,
                                                                      pos_diff, distance
                    append!(neighbors[particle], neighbor)
                end

                # All of these tests are designed to yield the same neighbor lists.
                # Note that we have to sort the neighbor lists because neighborhood searches
                # might produce different orders.
                @test sort(neighbors[1]) == [1, 3, 5]
                @test sort(neighbors[2]) == [2]
                @test sort(neighbors[3]) == [1, 3]
                @test sort(neighbors[4]) == [4]
                @test sort(neighbors[5]) == [1, 5]
            end
        end
    end
end;
