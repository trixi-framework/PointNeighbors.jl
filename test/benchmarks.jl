# Check that all benchmarks are running without errors.
# Note that these are only smoke tests, not verifying the result.
# Also note that these tests are run without coverage checks, since we want to
# cover everything with unit tests.
@testset verbose=true "Benchmarks" begin
    include("../benchmarks/benchmarks.jl")

    @testset verbose=true "`point_cloud`" begin
        cloud_sizes = [(10,), (10, 11), (9, 10, 7)]
        eltypes = (Float32, Float64)
        rotation_angles = (0, 10)

        name(size, ELTYPE, angle) = "$(length(size))D, $ELTYPE, Rotated by $(angle)°"
        @testset "$(name(cloud_size, ELTYPE, rotation_angle))" for cloud_size in
                                                                   cloud_sizes,
                                                                   ELTYPE in eltypes,
                                                                   rotation_angle in
                                                                   rotation_angles

            # The element type of the coordinates must be the type of `search_radius`,
            # also when the lattice is rotated.
            search_radius = ELTYPE(2.5)
            coords = point_cloud(cloud_size, search_radius; rotation_angle,
                                 perturbation_factor_position = 0)

            @test eltype(coords) == ELTYPE
            @test size(coords) == (length(cloud_size), prod(cloud_size))

            # The bounding box of the unperturbed lattice starts at 1 and ends at
            # `point_cloud_size`.
            NDIMS = length(cloud_size)
            # Note that we compare element-wise, since `isapprox` for arrays compares norms.
            atol = 100 * eps(ELTYPE)
            @test all(isapprox.(minimum(coords, dims = 2), 1; atol))
            @test all(isapprox.(vec(maximum(coords, dims = 2)),
                                point_cloud_size(cloud_size; rotation_angle); atol))

            # The first lattice axis is rotated by the rotation matrix.
            rotation = rotation_matrix(rotation_angle, Val(NDIMS))
            @test all(isapprox.(coords[:, 2] - coords[:, 1], rotation[:, 1]; atol))

            # The rotation preserves distances between points.
            lattice = point_cloud(cloud_size, search_radius;
                                  perturbation_factor_position = 0)
            distances(x) = [sqrt(sum(abs2, x[:, i] - x[:, j]))
                            for i in axes(x, 2), j in axes(x, 2)]
            @test all(isapprox.(distances(coords), distances(lattice); atol))
        end
    end

    @testset verbose=true "`almost_sorted_permutation`" begin
        @testset verbose=true "$(length(size))D" for size in [(50,), (10, 10), (5, 5, 5)]
            NDIMS = length(size)
            search_radius = 3.0
            coords = point_cloud(size, search_radius; rotation_angle = 10)
            n_points = Base.size(coords, 2)

            grid_nhs = GridNeighborhoodSearch{NDIMS}(; search_radius, n_points)
            cells(x, nhs) = [PointNeighbors.cell_coords(SVector{NDIMS}(x[:, i]), nhs)
                             for i in axes(x, 2)]

            @testset "Without displacement" begin
                permutation = almost_sorted_permutation(coords, grid_nhs, 0)

                # Without displacement, the points are sorted by cell.
                @test isperm(permutation)
                @test issorted(cells(coords[:, permutation], grid_nhs))
            end

            @testset "With displacement" begin
                permutation = almost_sorted_permutation(coords, grid_nhs, 0.3)
                @test isperm(permutation)

                # With displacement, the points are sorted by the cells of the
                # displaced coordinates, so they are not sorted by cell anymore.
                # In 1D, the displacement doesn't change the order of the cells.
                if NDIMS > 1
                    @test !issorted(cells(coords[:, permutation], grid_nhs))
                end
            end

            # `PrecomputedNeighborhoodSearch` and NHS implementations without a grid
            # must yield the same permutation as the `GridNeighborhoodSearch`.
            precomputed_nhs = PrecomputedNeighborhoodSearch{NDIMS}(; search_radius,
                                                                   n_points)
            trivial_nhs = TrivialNeighborhoodSearch{NDIMS}(; search_radius,
                                                           eachpoint = 1:n_points)
            @testset "$(nameof(typeof(nhs)))" for nhs in (precomputed_nhs, trivial_nhs)
                for sort_displacement in (0, 0.3)
                    @test almost_sorted_permutation(coords, nhs, sort_displacement) ==
                          almost_sorted_permutation(coords, grid_nhs, sort_displacement)
                end
            end
        end
    end

    @testset verbose=true "$(length(size))D" for size in [(50,), (10, 10), (5, 5, 5)]
        @testset verbose=true "`benchmark_count_neighbors`" begin
            @trixi_test_nowarn run_benchmark_default(benchmark_count_neighbors, size, 2)
            @trixi_test_nowarn run_benchmark_gpu(benchmark_count_neighbors, size, 2)
            @trixi_test_nowarn run_benchmark_full_grid(benchmark_count_neighbors, size, 2)
            @trixi_test_nowarn run_benchmark_precomputed(benchmark_count_neighbors, size,
                                                         2)
        end

        @testset verbose=true "`benchmark_n_body`" begin
            @trixi_test_nowarn run_benchmark_default(benchmark_n_body, size, 2)
            @trixi_test_nowarn run_benchmark_gpu(benchmark_n_body, size, 2)
        end

        @testset verbose=true "`benchmark_wcsph`" begin
            @trixi_test_nowarn run_benchmark_default(benchmark_wcsph, size, 2)
            @trixi_test_nowarn run_benchmark_gpu(benchmark_wcsph, size, 2)
        end

        @testset verbose=true "`benchmark_wcsph` Float32" begin
            @trixi_test_nowarn run_benchmark_default(benchmark_wcsph, size, 2,
                                                     search_radius_factor = 3.0f0)
            @trixi_test_nowarn run_benchmark_gpu(benchmark_wcsph, size, 2,
                                                 search_radius_factor = 3.0f0)
        end

        @testset verbose=true "`benchmark_tlsph`" begin
            @trixi_test_nowarn run_benchmark_default(benchmark_tlsph, size, 2)
            @trixi_test_nowarn run_benchmark_gpu(benchmark_tlsph, size, 2)
        end

        @testset verbose=true "`benchmark_tlsph_deformation_grad`" begin
            @trixi_test_nowarn run_benchmark_default(benchmark_tlsph_deformation_grad,
                                                     size, 2)
            @trixi_test_nowarn run_benchmark_gpu(benchmark_tlsph_deformation_grad, size, 2)
        end

        @testset verbose=true "`benchmark_initialize`" begin
            @trixi_test_nowarn run_benchmark_default(benchmark_initialize, size, 2)
            @trixi_test_nowarn run_benchmark_gpu(benchmark_initialize, size, 2)
        end

        @testset verbose=true "`benchmark_update_alternating`" begin
            @trixi_test_nowarn run_benchmark_default(benchmark_update_alternating, size, 2)
            @trixi_test_nowarn run_benchmark_gpu(benchmark_update_alternating, size, 2)
        end

        @testset verbose=true "`run_benchmark_updates`" begin
            @trixi_test_nowarn run_benchmark_updates(size, 2)
        end

        @testset verbose=true "`run_benchmark` options" begin
            NDIMS = length(size)
            neighborhood_searches = [TrivialNeighborhoodSearch{NDIMS}(),
                GridNeighborhoodSearch{NDIMS}()]

            # A neighborhood search without a grid as first NHS, which is used for sorting.
            @trixi_test_nowarn run_benchmark(benchmark_count_neighbors, size, 2,
                                             neighborhood_searches)
            @trixi_test_nowarn run_benchmark(benchmark_count_neighbors, size, 2,
                                             neighborhood_searches; shuffle = true)
            @trixi_test_nowarn run_benchmark(benchmark_count_neighbors, size, 2,
                                             neighborhood_searches; sort_displacement = 0)
        end

        @testset verbose=true "`max_neighbors` is passed through" begin
            # Two neighbors are not enough for any of the point clouds.
            for run_benchmark_ in (run_benchmark_default, run_benchmark_gpu,
                                   run_benchmark_precomputed)
                @test_throws ErrorException run_benchmark_(benchmark_count_neighbors,
                                                           size, 1; max_neighbors = 2)
            end
            @test_throws ErrorException run_benchmark_updates(size, 1; max_neighbors = 2)
        end
    end
end;
