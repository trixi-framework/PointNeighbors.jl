const POINTNEIGHBORS_TEST_ = lowercase(get(ENV, "POINTNEIGHBORS_TEST", "all"))

if POINTNEIGHBORS_TEST_ == "cuda"
    using CUDA
    CUDA.versioninfo()
    parallelization_backend = CUDABackend()
    supports_double_precision = true
    fp64_fastdiv = true
elseif POINTNEIGHBORS_TEST_ == "amdgpu"
    using AMDGPU
    AMDGPU.versioninfo()
    parallelization_backend = ROCBackend()
    supports_double_precision = true
    fp64_fastdiv = false
elseif POINTNEIGHBORS_TEST_ == "metal"
    using Metal
    Metal.versioninfo()
    parallelization_backend = MetalBackend()
    supports_double_precision = false
elseif POINTNEIGHBORS_TEST_ == "oneapi"
    using oneAPI
    oneAPI.versioninfo()
    parallelization_backend = oneAPIBackend()
    # The runners are using an iGPU, which does not support double precision
    supports_double_precision = false
else
    error("Unknown GPU backend: $POINTNEIGHBORS_TEST_")
end

@testset verbose=true "GPU tutorial" begin
    @trixi_test_nowarn trixi_include(joinpath(@__DIR__, "..", "docs", "literate", "src",
                                              "tut_gpu_usage.jl"),
                                     backend = parallelization_backend)
    @test n_neighbors_gpu isa PointNeighbors.AbstractGPUArray
    @test extrema(n_neighbors_gpu) == (11, 29)
end

@testset verbose=true "TreeNeighborhoodSearch" begin
    ELTYPES = supports_double_precision ? (Float32, Float64) : (Float32,)

    @testset "Tree Structure $ELTYPE" for ELTYPE in ELTYPES
        # Random points in 3D with search radii growing in x-direction
        coords = rand(ELTYPE, 3, 20_000)
        radii = ELTYPE(0.02) .+ ELTYPE(0.1) .* coords[1, :] .^ 2

        min_corner = zeros(ELTYPE, 3)
        max_corner = ones(ELTYPE, 3)
        cell_list = TreeCellList(; min_corner, max_corner, max_level = 5,
                                 max_points_per_cell = 500)
        nhs = TreeNeighborhoodSearch{3}(; cell_list, n_points = size(coords, 2),
                                        update_strategy = SerialUpdate())
        initialize_tree!(nhs, coords, radii)

        # Build the same tree on the GPU
        # Build the same tree on the GPU with the parallel update
        nhs_parallel = TreeNeighborhoodSearch{3}(; n_points = size(coords, 2),
                                                 cell_list = TreeCellList(; min_corner,
                                                                          max_corner,
                                                                          max_level = 5,
                                                                          max_points_per_cell = 500),
                                                 update_strategy = ParallelUpdate())
        nhs_gpu = PointNeighbors.Adapt.adapt(parallelization_backend, nhs_parallel)
        coords_gpu = PointNeighbors.Adapt.adapt(parallelization_backend, coords)
        radii_gpu = PointNeighbors.Adapt.adapt(parallelization_backend, radii)
        initialize_tree!(nhs_gpu, coords_gpu, radii_gpu; parallelization_backend)

        @test nhs_gpu.cell_list.leaf_level isa PointNeighbors.AbstractGPUArray
        @test nhs_gpu.update_strategy == ParallelUpdate()

        # The tree structure and the assignment of lists to leaves must be identical
        cell_list_gpu = PointNeighbors.Adapt.adapt(Array, nhs_gpu.cell_list)
        (; cell_list) = nhs
        @test cell_list_gpu.leaf_level == cell_list.leaf_level
        @test cell_list_gpu.n_points == cell_list.n_points
        @test cell_list_gpu.max_radius == cell_list.max_radius
        @test cell_list_gpu.leaves == cell_list.leaves

        # The order of the points within each list can be different
        @test all(0:(length(cell_list.leaves) - 1)) do cell
            points_gpu = PointNeighbors.points_in_leaf(cell_list_gpu, cell)
            sort(points_gpu) == sort(PointNeighbors.points_in_leaf(cell_list, cell))
        end
    end

    @testset "Neighbor Counts $ELTYPE, $(nameof(typeof(neighbor_criterion))), Pruning = $pruning, Periodic = $periodic" for ELTYPE in
                                                                                                                            ELTYPES,
                                                                                                                            neighbor_criterion in
                                                                                                                            (MaximumRadius(),
                                                                                                                             AverageRadius(),
                                                                                                                             QueryRadius()),
                                                                                                                            pruning in
                                                                                                                            (false,
                                                                                                                             true),
                                                                                                                            periodic in
                                                                                                                            (false,
                                                                                                                             true)
        # Points on a lattice with spacing 1/32 and search radii that are multiples of
        # 1/32. All distances and radii are exactly representable, so that the results
        # on the CPU and the GPU are identical, even for pairs with a distance of exactly
        # the average radius.
        lattice = CartesianIndices((32, 32, 32))
        coords = ELTYPE.(reduce(hcat,
                                [collect(Tuple(cell) .- 1) ./ 32
                                 for cell in vec(lattice)]))

        # Radii between 2/32 and 4/32 growing in x-direction, with a jump to 6/32
        # in the upper half of the domain
        radii = [ELTYPE(coords[3, i] < 0.5 ? (2 + round(2 * coords[1, i])) / 32 : 6 / 32)
                 for i in axes(coords, 2)]

        min_corner = zeros(ELTYPE, 3)
        max_corner = ones(ELTYPE, 3)
        # With periodicity, use 4 root cells in each dimension, which are larger than
        # all search radii
        if periodic
            periodic_box = PeriodicBox(; min_corner, max_corner)
            cell_list = TreeCellList(; min_corner, max_corner, max_level = 2,
                                     root_cell_size = ELTYPE(0.25),
                                     max_points_per_cell = size(coords, 2))
        else
            periodic_box = nothing
            cell_list = TreeCellList(; min_corner, max_corner, max_level = 4,
                                     max_points_per_cell = size(coords, 2))
        end
        nhs = TreeNeighborhoodSearch{3}(; cell_list, n_points = size(coords, 2),
                                        neighbor_criterion, pruning, periodic_box)
        initialize_tree!(nhs, coords, radii)

        # Count neighbors of all points with their own search radius
        count_neighbor(point, neighbor, pos_diff, distance) = Int32(1)

        function count_neighbors!(n_neighbors, coords, radii, nhs, parallelization_backend)
            PointNeighbors.@threaded parallelization_backend for point in axes(coords, 2)
                @inbounds n_neighbors[point] = mapreduce_neighbor(count_neighbor, +,
                                                                  coords, coords, nhs,
                                                                  point; init = Int32(0),
                                                                  search_radius = radii[point])
            end

            return n_neighbors
        end

        n_neighbors = count_neighbors!(zeros(Int32, size(coords, 2)), coords, radii, nhs,
                                       SerialBackend())

        nhs_gpu = PointNeighbors.Adapt.adapt(parallelization_backend,
                                             copy_neighborhood_search(nhs, 0,
                                                                      size(coords, 2)))
        coords_gpu = PointNeighbors.Adapt.adapt(parallelization_backend, coords)
        radii_gpu = PointNeighbors.Adapt.adapt(parallelization_backend, radii)
        initialize_tree!(nhs_gpu, coords_gpu, radii_gpu; parallelization_backend)

        n_neighbors_gpu = PointNeighbors.Adapt.adapt(parallelization_backend,
                                                     zeros(Int32, size(coords, 2)))
        count_neighbors!(n_neighbors_gpu, coords_gpu, radii_gpu, nhs_gpu,
                         parallelization_backend)

        @test Array(n_neighbors_gpu) == n_neighbors
    end
end
