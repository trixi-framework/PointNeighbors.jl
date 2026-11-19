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

@testset verbose=true "Sorting on the GPU: `FullGridCellList` with `CompactVectorOfVectors`" begin
    ELTYPE = supports_double_precision ? Float64 : Float32
    search_radius = convert(ELTYPE, 2.5)
    coords = convert.(ELTYPE, point_cloud((7, 6), search_radius, shuffle = true))
    n_points = size(coords, 2)

    min_corner = minimum(coords, dims = 2)
    max_corner = maximum(coords, dims = 2)
    cell_list = FullGridCellList(; min_corner, max_corner, search_radius,
                                 backend = CompactVectorOfVectors{Int32})
    nhs = GridNeighborhoodSearch{2}(; search_radius, n_points, cell_list)

    coords_gpu = Adapt.adapt(parallelization_backend, coords)
    nhs_gpu = Adapt.adapt(parallelization_backend, nhs)

    # Every other point is inactive.
    eachindex_y = 1:2:n_points
    eachindex_y_gpu = Adapt.adapt(parallelization_backend, collect(eachindex_y))

    # Now initialize and compare the cell lists on CPU and GPU.
    initialize!(nhs, coords, coords; eachindex_y)
    initialize!(nhs_gpu, coords_gpu, coords_gpu; eachindex_y = eachindex_y_gpu,
                parallelization_backend)

    @test nhs_gpu.cell_list.cells.values isa PointNeighbors.AbstractGPUArray

    # The order inside each cell is not deterministic, so we compare sorted cells.
    cells = nhs.cell_list.cells
    cells_gpu = Adapt.adapt(Array, nhs_gpu.cell_list.cells)
    @test cells_gpu.first_bin_index == cells.first_bin_index
    @test all(sort(cells_gpu[cell]) == sort(cells[cell]) for cell in eachindex(cells))

    # The inactive points are stored in increasing order after all cells.
    n_active = length(eachindex_y)
    @test cells_gpu.values[(n_active + 1):end] == cells.values[(n_active + 1):end]

    # Count the neighbors on the GPU.
    n_neighbors_gpu = Adapt.adapt(parallelization_backend, zeros(Int, n_points))
    foreach_point(coords_gpu, coords_gpu, nhs_gpu) do point, neighbors
        n_neighbors = mapreduce_neighbor((point, neighbor, pos_diff, distance) -> 1,
                                         +, neighbors; init = 0)
        @inbounds n_neighbors_gpu[point] = n_neighbors
    end

    n_neighbors_expected = [count(j -> sum(abs2, coords[:, i] - coords[:, j]) <=
                                       search_radius^2, eachindex_y)
                            for i in axes(coords, 2)]
    @test Array(n_neighbors_gpu) == n_neighbors_expected
end
