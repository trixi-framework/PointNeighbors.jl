# # [GPU Usage](@id tut_gpu_usage)

# This tutorial shows how to use PointNeighbors.jl on GPUs.
# We adapt the [basic usage tutorial](@ref tut_basic_usage) to run on GPUs.
using PointNeighbors
using Adapt

# PointNeighbors.jl provides vendor-agnostic GPU support through KernelAbstractions.jl.
# Load the appropriate package for your GPU and define the corresponding backend:
# - `using CUDA; backend = CUDABackend()` for NVIDIA GPUs with CUDA.jl
# - `using AMDGPU; backend = ROCBackend()` for AMD GPUs with AMDGPU.jl
# - `using Metal; backend = MetalBackend()` for Apple Silicon GPUs with Metal.jl
# - `using oneAPI; backend = oneAPIBackend()` for Intel GPUs with oneAPI.jl
import KernelAbstractions # hide
backend = KernelAbstractions.CPU() # hide
nothing # hide

# ## Create coordinates on the CPU

# We create the same regular grid of points in 2D as in the basic usage tutorial.
# For better GPU performance, we use `Float32` instead of `Float64`.
# Note that some GPUs (notably Apple Silicon) do not support `Float64` at all.
n_points_per_dimension = (100, 100)
n_points = prod(n_points_per_dimension)
coordinates = Array{Float32}(undef, 2, n_points)
cartesian_indices = CartesianIndices(n_points_per_dimension)

for i in axes(coordinates, 2)
    coordinates[:, i] .= Tuple(cartesian_indices[i])
end

# We can use Adapt.jl to move this coordinates array to the GPU.
coordinates_gpu = adapt(backend, coordinates)

# ## Create and initialize the neighborhood search

# After taking computing the difference between coordinates of neighboring particles,
# [`foreach_point_neighbor`](@ref) converts the result to the type of `search_radius` before
# computing the distance. The type of `search_radius` therefore determines the types of
# `pos_diff` and `distance` inside `foreach_point_neighbor`. We need to make sure to choose
# the data type that we want to use for our computations, which will usually be `Float32`
# when working on GPUs.
#
# Note that it is common in Smoothed Particle Hydrodynamics to store the coordinates
# in `Float64` and everything else in `Float32` to avoid large errors in the distance
# computations. This is also supported by PointNeighbors.jl. Using `Float64` coordinates
# with a `Float32` search radius limits `Float64` operations by converting to `Float32`
# after loading the point coordinates and computing the difference.
# `pos_diff` and `distance` inside `foreach_point_neighbor` will all use `Float32`.
search_radius = 3.0f0
nothing # hide

# For GPU compatibility, we need to use a [`FullGridCellList`](@ref), which is optimized
# for maximum performance, but requires a rectangular bounding box for the domain,
# as opposed to the default [`DictionaryCellList`](@ref) that supports potentially
# infinite domains.
min_corner = minimum(coordinates, dims = 2)
max_corner = maximum(coordinates, dims = 2)
cell_list = FullGridCellList(; search_radius, min_corner, max_corner)
nothing # hide

# Now we can create the neighborhood search as usual.
nhs = GridNeighborhoodSearch{2}(; search_radius, cell_list, n_points)
nothing # hide

# This neighborhood search is currently living in CPU memory, so we first need to move it
# to the GPU. We can also use Adapt.jl for this.
nhs_gpu = adapt(backend, nhs)
nothing # hide

# From now on, everything happens on the GPU, so we need to use the GPU coordinates.
initialize!(nhs_gpu, coordinates_gpu, coordinates_gpu; parallelization_backend = backend)
nothing # hide

# ## Count neighbors on the GPU

# We can now use the same function as in the [basic usage tutorial](@ref tut_basic_usage).
# The parallelization backend is detected automatically from the type of the coordinates.
n_neighbors_gpu = adapt(backend, zeros(Int, n_points))

function count_neighbors!(n_neighbors, coordinates, nhs)
    n_neighbors .= 0

    foreach_point_neighbor(coordinates, coordinates, nhs) do i, j, pos_diff, distance
        ## `foreach_point_neighbor` makes sure that `i` and `j` are in bounds
        ## of the respective coordinates.
        ## We are now inside a GPU kernel, so using `@inbounds` is important for performance.
        @inbounds n_neighbors[i] += 1
    end

    return n_neighbors
end

# Now we can run the neighbor loop on the GPU.
# We just need to make sure to pass the GPU coordinates, GPU neighborhood search,
# and GPU neighbor count array.
count_neighbors!(n_neighbors_gpu, coordinates_gpu, nhs_gpu)
extrema(n_neighbors_gpu)
