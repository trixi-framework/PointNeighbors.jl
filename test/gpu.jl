using Adapt: Adapt

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

@testset verbose=true "Sorting points with `CompactVectorOfVectors`" begin
    ELTYPE = supports_double_precision ? Float64 : Float32
    search_radius = convert(ELTYPE, 2.5)
    coords = convert.(ELTYPE, point_cloud((30, 31), search_radius, shuffle = true))
    n_points = size(coords, 2)

    min_corner = minimum(coords, dims = 2)
    max_corner = maximum(coords, dims = 2)
    cell_list = FullGridCellList(; min_corner, max_corner, search_radius,
                                 backend = CompactVectorOfVectors{Int32})
    nhs = GridNeighborhoodSearch{2}(; search_radius, n_points, cell_list)

    coords_gpu = Adapt.adapt(parallelization_backend, coords)
    nhs_gpu = Adapt.adapt(parallelization_backend, nhs)

    # Every other point is inactive
    eachindex_y = 1:2:n_points
    eachindex_y_gpu = Adapt.adapt(parallelization_backend, collect(eachindex_y))

    initialize!(nhs, coords, coords; eachindex_y)
    initialize!(nhs_gpu, coords_gpu, coords_gpu; eachindex_y = eachindex_y_gpu,
                parallelization_backend)

    # The sorting is deterministic, so the permutations must be identical
    @test point_permutation(nhs_gpu) isa PointNeighbors.AbstractGPUArray
    @test Array(point_permutation(nhs_gpu)) == point_permutation(nhs)

    # Count the neighbors on the GPU with the sorted points
    perm = point_permutation(nhs)
    sorted_coords_gpu = Adapt.adapt(parallelization_backend, coords[:, perm])
    sorted_nhs_gpu = sorted_neighborhood_search(nhs_gpu)

    n_neighbors_gpu = Adapt.adapt(parallelization_backend, zeros(Int, n_points))
    foreach_point(coords_gpu, sorted_coords_gpu, sorted_nhs_gpu) do point, neighbors
        n_neighbors = mapreduce_neighbor((point, neighbor, pos_diff, distance) -> 1,
                                         +, neighbors; init = 0)
        @inbounds n_neighbors_gpu[point] = n_neighbors
    end

    n_neighbors_expected = [count(j -> sum(abs2, coords[:, i] - coords[:, j]) <=
                                       search_radius^2, eachindex_y)
                            for i in axes(coords, 2)]
    @test Array(n_neighbors_gpu) == n_neighbors_expected
end
