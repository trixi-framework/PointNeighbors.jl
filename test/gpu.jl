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

@testset verbose=true "`update!` cell overflow" begin
    to_gpu(x) = PointNeighbors.Adapt.adapt(parallelization_backend, x)

    cell_list = FullGridCellList(; search_radius = 1.0f0, min_corner = (0.0f0, 0.0f0),
                                 max_corner = (10.0f0, 10.0f0), max_points_per_cell = 9)
    nhs = GridNeighborhoodSearch{2}(; cell_list, search_radius = 1.0f0, n_points = 10)
    nhs_gpu = to_gpu(nhs)

    # Spread points evenly on a line from (0.5, 0.5) to (9.5, 9.5).
    y = to_gpu([0.5f0 + i for _ in 1:2, i in 0:9])
    @trixi_test_nowarn initialize!(nhs_gpu, y, y)

    # Move all points into the same cell.
    # The error message is not propagated from GPU kernels.
    y2 = to_gpu(fill(5.5f0, 2, 10))
    @test_throws Exception update!(nhs_gpu, y, y2)
end
