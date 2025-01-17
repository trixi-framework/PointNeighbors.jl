module PointNeighborsCUDAExt

using PointNeighbors: PointNeighbors, CUDAMultiGPUBackend, DynamicVectorOfVectors,
                      GridNeighborhoodSearch, FullGridCellList, extract_svector,
                      cell_coords, cell_index, check_cell_bounds, pushat_atomic!
using CUDA: CUDA, CuArray, CUDABackend, cu
using UnsafeAtomics: UnsafeAtomics
using PointNeighbors.KernelAbstractions: KernelAbstractions, @kernel, @index
using PointNeighbors.Adapt: Adapt

const UnifiedCuArray = CuArray{<:Any, <:Any, CUDA.UnifiedMemory}

# This is needed because TrixiParticles passes `get_backend(coords)` to distinguish between
# `nothing` (Polyester.jl) and `KernelAbstractions.CPU`.
PointNeighbors.get_backend(x::UnifiedCuArray) = CUDAMultiGPUBackend()

# Convert input array to `CuArray` with unified memory
function Adapt.adapt_structure(to::CUDAMultiGPUBackend, array::Array)
    return CuArray{eltype(array), ndims(array), CUDA.UnifiedMemory}(array)
end

@inline function PointNeighbors.parallel_foreach(f, iterator, x::UnifiedCuArray)
    PointNeighbors.parallel_foreach(f, iterator, CUDAMultiGPUBackend())
end

@inline function PointNeighbors.parallel_foreach(f::T, iterator, x::CUDAMultiGPUBackend) where {T}
    parallel_foreach2((i, gpu) -> @inline f(i), iterator, x)
end

@inline function parallel_foreach2(f, iterator, x::UnifiedCuArray)
    parallel_foreach2(f, iterator, CUDAMultiGPUBackend())
end

KernelAbstractions.@kernel function generic_kernel2(f, gpu)
    i = KernelAbstractions.@index(Global)
    @inline f(i, gpu)
end

@inline function parallel_foreach2(f, iterator, x::CUDAMultiGPUBackend)
    # On the GPU, we can only loop over `1:N`. Therefore, we loop over `1:length(iterator)`
    # and index with `iterator[eachindex(iterator)[i]]`.
    # Note that this only works with vector-like iterators that support arbitrary indexing.
    indices = eachindex(iterator)

    # Skip empty loops
    length(indices) == 0 && return

    # Partition `ndrange` to the GPUs
    n_gpus = length(CUDA.devices())
    indices_split = Iterators.partition(indices, ceil(Int, length(indices) / n_gpus))
    @assert length(indices_split) <= n_gpus

    backend = CUDABackend()

    # Synchronize each device
    for i in 1:n_gpus
        CUDA.device!(i - 1)
        KernelAbstractions.synchronize(backend)
    end

    # Launch kernel on each device
    for (i, indices_) in enumerate(indices_split)
        # Select the correct device for this partition
        CUDA.device!(i - 1)

        # Call the generic kernel, which only calls a function with the global GPU index
        generic_kernel2(backend)(i, ndrange = length(indices_)) do j, gpu
            @inbounds @inline f(iterator[indices_[j]], gpu)
        end
    end

    # Synchronize each device
    for i in 1:n_gpus
        CUDA.device!(i - 1)
        KernelAbstractions.synchronize(backend)
    end

    # Select first device again
    CUDA.device!(0)
end

struct MultiGPUVectorOfVectors{T, VOV, V} <: AbstractVector{Array{T, 1}}
    vov::VOV
    vov_per_gpu::V
end

function MultiGPUVectorOfVectors(vov, vov_per_gpu)
    MultiGPUVectorOfVectors{eltype(vov.backend), typeof(vov), typeof(vov_per_gpu)}(vov, vov_per_gpu)
end

Adapt.@adapt_structure(MultiGPUVectorOfVectors)

function Adapt.adapt_structure(to::CUDAMultiGPUBackend, vov::DynamicVectorOfVectors{T}) where {T}
    max_inner_length, max_outer_length = size(vov.backend)

    n_gpus = length(CUDA.devices())
    vov_per_gpu = [DynamicVectorOfVectors{T}(; max_outer_length, max_inner_length) for _ in 1:n_gpus]

    vov_ = DynamicVectorOfVectors(Adapt.adapt(to, vov.backend), Adapt.adapt(to, vov.length_), Adapt.adapt(to, vov.lengths))
    vov_per_gpu_ = ntuple(i -> Adapt.adapt(CuArray, vov_per_gpu[i]), n_gpus)
    for vov__ in vov_per_gpu_
        vov__.length_[] = vov.length_[]
    end

    return MultiGPUVectorOfVectors{T, typeof(vov_), typeof(vov_per_gpu_)}(vov_, vov_per_gpu_)
end

const MultiGPUNeighborhoodSearch = GridNeighborhoodSearch{<:Any, <:Any, <:FullGridCellList{<:MultiGPUVectorOfVectors}}

function PointNeighbors.initialize_grid!(neighborhood_search::MultiGPUNeighborhoodSearch, y::AbstractMatrix)
    (; cell_list) = neighborhood_search
    (; cells) = cell_list
    (; vov, vov_per_gpu) = cells

    for vov_ in vov_per_gpu
        vov_.lengths .= 0
    end

    if neighborhood_search.search_radius < eps()
        # Cannot initialize with zero search radius.
        # This is used in TrixiParticles when a neighborhood search is not used.
        return neighborhood_search
    end

    # Fill cell lists per GPU
    # TODO split range into chunks for each GPU
    parallel_foreach2(axes(y, 2), y) do point, gpu
        # Get cell index of the point's cell
        point_coords = extract_svector(y, Val(ndims(neighborhood_search)), point)
        cell = cell_coords(point_coords, neighborhood_search)

        # Add point to corresponding cell
        @boundscheck check_cell_bounds(cell_list, cell)

        # The atomic version of `pushat!` uses atomics to avoid race conditions when
        # `pushat!` is used in a parallel loop.
        @inbounds pushat_atomic!(vov_per_gpu[gpu], cell_index(cell_list, cell), point)
    end

    lengths = ntuple(gpu -> vov_per_gpu[gpu].lengths, length(vov_per_gpu))
    CUDA.synchronize()
    offsets_ = cu(cumsum(lengths), unified = true)
    CUDA.synchronize()
    vov.lengths .= offsets_[end]
    CUDA.synchronize()
    offsets = offsets_ .- lengths

    # Merge cell lists
    parallel_foreach2(axes(vov, 2), y) do cell, gpu
        offset = offsets[gpu][cell]

        points = vov_per_gpu[gpu][cell]
        for i in eachindex(points)
            vov.backend[offset + i, cell] = points[i]
        end
    end

    return neighborhood_search
end

function PointNeighbors.update!(neighborhood_search::MultiGPUNeighborhoodSearch,
                                x::AbstractMatrix, y::AbstractMatrix;
                                points_moving = (true, true), parallelization_backend = x)
    # The coordinates of the first set of points are irrelevant for this NHS.
    # Only update when the second set is moving.
    points_moving[2] || return neighborhood_search

    PointNeighbors.initialize_grid!(neighborhood_search, y)
end

end # module
