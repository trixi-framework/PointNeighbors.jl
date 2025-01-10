module PointNeighborsCUDAExt

using PointNeighbors: PointNeighbors, generic_kernel, CUDAMultiGPUBackend, KernelAbstractions
using CUDA: CUDA, CuArray, CUDABackend

const UnifiedCuArray = CuArray{<:Any, <:Any, CUDA.UnifiedMemory}

# This is needed because TrixiParticles passes `get_backend(coords)` to distinguish between
# `nothing` (Polyester.jl) and `KernelAbstractions.CPU`.
PointNeighbors.get_backend(x::UnifiedCuArray) = CUDAMultiGPUBackend()

# Convert input array to `CuArray` with unified memory
function PointNeighbors.Adapt.adapt_structure(to::CUDAMultiGPUBackend, array::Array)
    return CuArray{eltype(array), ndims(array), CUDA.UnifiedMemory}(array)
end

@inline function PointNeighbors.parallel_foreach(f, iterator, x::UnifiedCuArray)
    PointNeighbors.parallel_foreach(f, iterator, CUDAMultiGPUBackend())
end

# On GPUs, execute `f` inside a GPU kernel with KernelAbstractions.jl
@inline function PointNeighbors.parallel_foreach(f, iterator, x::CUDAMultiGPUBackend)
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

    # Spawn kernel on each device
    for (i, indices_) in enumerate(indices_split)
        # Select the correct device for this partition
        CUDA.device!(i - 1)

        # Call the generic kernel, which only calls a function with the global GPU index
        generic_kernel(backend)(ndrange = length(indices_)) do j
            @inbounds @inline f(iterator[indices_[j]])
        end
    end

    # Synchronize each device
    for i in 1:length(indices_split)
        CUDA.device!(i - 1)
        KernelAbstractions.synchronize(backend)
    end

    # Select first device again
    CUDA.device!(0)
end

end # module
