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
