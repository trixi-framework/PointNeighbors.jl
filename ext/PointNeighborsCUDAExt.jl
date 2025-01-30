module PointNeighborsCUDAExt

using PointNeighbors: PointNeighbors, CUDAMultiGPUBackend, DynamicVectorOfVectors,
                      GridNeighborhoodSearch, FullGridCellList, extract_svector,
                      cell_coords, cell_index, @threaded, generic_kernel
using CUDA: CUDA, CuArray, CUDABackend, cu
using PointNeighbors.KernelAbstractions: KernelAbstractions, @kernel, @index
using PointNeighbors.Adapt: Adapt
using Base: @propagate_inbounds

# Define custom type wrapping a `CuArray` with unified memory to dispatch on it
struct MultiGPUArray{T, N, A} <: AbstractArray{T, N}
    array::A

    function MultiGPUArray(array::AbstractArray{T, N}) where {T, N}
        new{T, N, typeof(array)}(array)
    end
end

Adapt.@adapt_structure MultiGPUArray

function Base.similar(m::MultiGPUArray, ::Type{T}) where {T}
    array = similar(m.array, T)
    return MultiGPUArray(array)
end

Base.parent(m::MultiGPUArray) = m.array
Base.size(m::MultiGPUArray) = size(m.array)
Base.getindex(m::MultiGPUArray, i...) = getindex(m.array, i...)
Base.setindex!(m::MultiGPUArray, x...) = (setindex!(m.array, x...); m)

@inline function Base.fill!(m::MultiGPUArray, x)
    # TODO check bounds
    # TODO convert x
    @threaded m for i in eachindex(m.array)
        @inbounds m.array[i] = x
    end

    return m
end

function Base.copyto!(dest::MultiGPUArray, src::MultiGPUArray)
    # TODO check bounds
    @threaded dest for i in eachindex(dest.array)
        @inbounds dest.array[i] = src.array[i]
    end

    return dest
end

function Base.copyto!(dest::MultiGPUArray, src::AbstractArray)
    copyto!(dest.array, src)
    return dest
end

function Base.copyto!(dest::MultiGPUArray, indices1::CartesianIndices, src::AbstractArray, indices2::CartesianIndices)
    copyto!(dest.array, indices1, src, indices2)
    return dest
end

function Base.copyto!(dest::AbstractArray, src::MultiGPUArray)
    copyto!(dest, src.array)
    return dest
end

function Base.Broadcast.BroadcastStyle(::Type{MultiGPUArray{T, N, A}}) where {T, N, A}
    Broadcast.BroadcastStyle(A)
end

Base.view(m::MultiGPUArray, i) = MultiGPUArray(view(m.array, i))
Base.reshape(m::MultiGPUArray, dims::Base.Dims) = MultiGPUArray(reshape(m.array, dims))

# This is needed because TrixiParticles passes `get_backend(coords)` to distinguish between
# `nothing` (Polyester.jl) and `KernelAbstractions.CPU`.
PointNeighbors.get_backend(::MultiGPUArray) = CUDAMultiGPUBackend()

# Convert input array to `MultiGPUArray` wrapping a `CuArray` with unified memory
function Adapt.adapt_structure(::CUDAMultiGPUBackend, array::Array)
    return MultiGPUArray(CuArray{eltype(array), ndims(array), CUDA.UnifiedMemory}(array))
end

@inline function PointNeighbors.parallel_foreach(f, iterator, ::MultiGPUArray)
    PointNeighbors.parallel_foreach(f, iterator, CUDAMultiGPUBackend())
end

@inline function PointNeighbors.parallel_foreach(f::T, iterator, ::CUDAMultiGPUBackend) where {T}
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
        generic_kernel(backend)(ndrange = length(indices_)) do j
            @inbounds @inline f(iterator[indices_[j]])
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

Base.any(f::Function, A::MultiGPUArray) = mapreduce(f, |, A, init = false)

function Base.mapreduce(f, op, A::MultiGPUArray; dims=:, init=nothing)
    n_gpus = length(CUDA.devices())
    indices = eachindex(A.array)

    # Skip empty loops
    if length(indices) == 0
        init !== nothing && return init
        throw(ArgumentError("reducing over an empty collection is not allowed; consider supplying `init` to the reducer"))
    end

    indices_split = Iterators.partition(indices, ceil(Int, length(indices) / n_gpus))
    @assert length(indices_split) <= n_gpus

    backend = CUDABackend()

    # Synchronize each device
    for i in 1:n_gpus
        CUDA.device!(i - 1)
        KernelAbstractions.synchronize(backend)
    end

    init2 = init === nothing ? Base._InitialValue() : init
    result = mapreduce(op, enumerate(indices_split), init=init2) do (i, indices_)
        # Select the correct device for this partition
        CUDA.device!(i - 1)

        # return mapreduce(@inline(i -> @inbounds f(A.array[i])), op, indices_)
        return mapreduce(f, op, view(A.array, indices_); dims=dims, init=init)
    end

    # Synchronize each device
    for i in 1:n_gpus
        CUDA.device!(i - 1)
        KernelAbstractions.synchronize(backend)
    end

    return result
end

# `@..` by FastBroadcast.jl forwards to this function
@inline function CUDA.GPUArrays._copyto!(dest::MultiGPUArray, bc::Broadcast.Broadcasted)
    axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
    isempty(dest) && return dest
    bc = Broadcast.preprocess(dest, bc)

    @threaded dest for i in eachindex(dest.array)
        @inbounds dest.array[i] = bc[i]
    end

    # @kernel function broadcast_kernel_linear(dest, bc)
    #     I = @index(Global, Linear)
    #     @inbounds dest[I] = bc[I]
    # end

    # @kernel function broadcast_kernel_cartesian(dest, bc)
    #     I = @index(Global, Cartesian)
    #     @inbounds dest[I] = bc[I]
    # end

    # broadcast_kernel = if ndims(dest) == 1 ||
    #                       (isa(IndexStyle(dest), IndexLinear) &&
    #                        isa(IndexStyle(bc), IndexLinear))
    #     broadcast_kernel_linear(get_backend(dest))
    # else
    #     broadcast_kernel_cartesian(get_backend(dest))
    # end

    # # ndims check for 0D support
    # broadcast_kernel(dest, bc; ndrange = ndims(dest) > 0 ? size(dest) : (1,))
    # if eltype(dest) <: BrokenBroadcast
    #     throw(ArgumentError("Broadcast operation resulting in $(eltype(eltype(dest))) is not GPU compatible"))
    # end

    return dest
end

############################################################################################
# First approach: Use actual system atomics on unified memory
############################################################################################
# function atomic_system_add(ptr::CUDA.LLVMPtr{Int32, CUDA.AS.Global}, val::Int32)
#     CUDA.LLVM.Interop.@asmcall(
#         "atom.sys.global.add.u32 \$0, [\$1], \$2;",
#         "=r,l,r,~{memory}",
#         true, Int32, Tuple{CUDA.LLVMPtr{Int32, CUDA.AS.Global}, Int32},
#         ptr, val
#     )
# end

# @inline function pushat_atomic_system!(vov, i, value)
#     (; backend, lengths) = vov

#     @boundscheck checkbounds(vov, i)

#     # Increment the column length with an atomic add to avoid race conditions.
#     # Store the new value since it might be changed immediately afterwards by another
#     # thread.
#     # new_length = Atomix.@atomic lengths[i] += 1
#     new_length = atomic_system_add(pointer(lengths, i), Int32(1)) + 1

#     # We can write here without race conditions, since the atomic add guarantees
#     # that `new_length` is different for each thread.
#     backend[new_length, i] = value

#     return vov
# end

# const MultiGPUNeighborhoodSearch = GridNeighborhoodSearch{<:Any, <:Any, <:FullGridCellList{<:DynamicVectorOfVectors{<:Any, <:CuArray{<:Any, <:Any, CUDA.UnifiedMemory}}}}

# function PointNeighbors.initialize_grid!(neighborhood_search::MultiGPUNeighborhoodSearch, y::AbstractMatrix)
#     (; cell_list) = neighborhood_search

#     cell_list.cells.lengths .= 0

#     if neighborhood_search.search_radius < eps()
#         # Cannot initialize with zero search radius.
#         # This is used in TrixiParticles when a neighborhood search is not used.
#         return neighborhood_search
#     end

#     # Faster on a single GPU
#     # CUDA.NVTX.@range "threaded loop" begin
#     @threaded y for point in axes(y, 2)
#         # Get cell index of the point's cell
#         point_coords = PointNeighbors.extract_svector(y, Val(ndims(neighborhood_search)), point)
#         cell = PointNeighbors.cell_coords(point_coords, neighborhood_search)

#         # Add point to corresponding cell
#         pushat_atomic_system!(cell_list.cells, PointNeighbors.cell_index(cell_list, cell), point)
#     end
# # end

#     return neighborhood_search
# end

# function PointNeighbors.update_grid!(neighborhood_search::MultiGPUNeighborhoodSearch,
#                       y::AbstractMatrix; parallelization_backend = y)
#     PointNeighbors.initialize_grid!(neighborhood_search, y)
# end

############################################################################################
# Second approach: Use a device array to update and copy to unified memory
############################################################################################
# struct MultiGPUVectorOfVectors{T, VU, VD} <: AbstractVector{Array{T, 1}}
#     vov_unified :: VU
#     vov_device  :: VD
# end

# function MultiGPUVectorOfVectors(vov_unified, vov_device)
#     MultiGPUVectorOfVectors{eltype(vov_unified.backend), typeof(vov_unified), typeof(vov_device)}(vov_unified, vov_device)
# end

# # Adapt.@adapt_structure(MultiGPUVectorOfVectors)

# function Adapt.adapt_structure(to::CUDA.KernelAdaptor, vov::MultiGPUVectorOfVectors)
#     return Adapt.adapt(to, vov.vov_unified)
# end

# @propagate_inbounds function Base.getindex(vov::MultiGPUVectorOfVectors, i)
#     return getindex(vov.vov_unified, i)
# end

# function Adapt.adapt_structure(to::CUDAMultiGPUBackend, vov::DynamicVectorOfVectors{T}) where {T}
#     max_inner_length, max_outer_length = size(vov.backend)

#     # Make sure the vector of vectors in device memory lives on the first GPU
#     CUDA.device!(0)

#     vov_unified = DynamicVectorOfVectors(Adapt.adapt(to, vov.backend), Adapt.adapt(to, vov.length_), Adapt.adapt(to, vov.lengths))
#     vov_device = Adapt.adapt(CuArray, vov)

#     return MultiGPUVectorOfVectors(vov_unified, vov_device)
# end

# const MultiGPUNeighborhoodSearch = GridNeighborhoodSearch{<:Any, <:Any, <:FullGridCellList{<:MultiGPUVectorOfVectors}}

# function PointNeighbors.initialize_grid!(neighborhood_search::MultiGPUNeighborhoodSearch, y::AbstractMatrix)
#     (; cell_list) = neighborhood_search
#     (; cells) = cell_list
#     (; vov_unified, vov_device) = cells

#     if neighborhood_search.search_radius < eps()
#         # Cannot initialize with zero search radius.
#         # This is used in TrixiParticles when a neighborhood search is not used.
#         return neighborhood_search
#     end

#     vov_device.lengths .= 0

#     CUDA.device!(0)

#     # Fill vector of vectors in device memory (on a single GPU)
#     @threaded CUDABackend() for point in axes(y, 2)
#         # Get cell index of the point's cell
#         point_coords = extract_svector(y, Val(ndims(neighborhood_search)), point)
#         cell = cell_coords(point_coords, neighborhood_search)

#         # Add point to corresponding cell
#         @boundscheck PointNeighbors.check_cell_bounds(cell_list, cell)

#         # The atomic version of `pushat!` uses atomics to avoid race conditions when
#         # `pushat!` is used in a parallel loop.
#         @inbounds PointNeighbors.pushat_atomic!(vov_device, cell_index(cell_list, cell), point)
#     end

#     # Copy vector of vectors to unified memory
#     vov_unified.backend .= vov_device.backend
#     vov_unified.lengths .= vov_device.lengths
#     vov_unified.length_[] = vov_device.length_[]

#     return neighborhood_search
# end

# function PointNeighbors.update!(neighborhood_search::MultiGPUNeighborhoodSearch,
#                                 x::AbstractMatrix, y::AbstractMatrix;
#                                 points_moving = (true, true), parallelization_backend = x)
#     # The coordinates of the first set of points are irrelevant for this NHS.
#     # Only update when the second set is moving.
#     points_moving[2] || return neighborhood_search

#     PointNeighbors.initialize_grid!(neighborhood_search, y)
# end

############################################################################################
# Third approach: Like second approach, but a device array for each GPU.

# This should have the advantage that each GPU updates the part of the grid that should
# already be in its memory and we can avoid moving everything to the first GPU.
############################################################################################
KernelAbstractions.@kernel function generic_kernel2(f, gpu)
    i = KernelAbstractions.@index(Global)
    @inline f(i, gpu)
end

@inline function parallel_foreach2(f::T, iterator, vov_per_gpu) where {T}
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
        generic_kernel2(backend)(vov_per_gpu[i], ndrange = length(indices_)) do j, vov_per_gpu
            @inbounds @inline f(iterator[indices_[j]], vov_per_gpu)
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
    vov_unified::VOV
    vov_per_gpu::V
end

function MultiGPUVectorOfVectors(vov_unified, vov_per_gpu)
    MultiGPUVectorOfVectors{eltype(vov_unified.backend), typeof(vov_unified), typeof(vov_per_gpu)}(vov_unified, vov_per_gpu)
end

# Adapt.@adapt_structure(MultiGPUVectorOfVectors)

function Adapt.adapt_structure(to::CUDA.KernelAdaptor, vov::MultiGPUVectorOfVectors)
    return Adapt.adapt(to, vov.vov_unified)
end

function Adapt.adapt_structure(to::CUDAMultiGPUBackend, vov::DynamicVectorOfVectors{T}) where {T}
    max_inner_length, max_outer_length = size(vov.backend)

    n_gpus = length(CUDA.devices())
    vov_per_gpu = [DynamicVectorOfVectors{T}(; max_outer_length, max_inner_length) for _ in 1:n_gpus]

    vov_unified = DynamicVectorOfVectors(Adapt.adapt(to, vov.backend), Adapt.adapt(to, vov.length_), Adapt.adapt(to, vov.lengths))
    vov_per_gpu_ = ntuple(i -> Adapt.adapt(CuArray, vov_per_gpu[i]), n_gpus)
    for vov__ in vov_per_gpu_
        vov__.length_[] = vov.length_[]
    end

    return MultiGPUVectorOfVectors{T, typeof(vov_unified), typeof(vov_per_gpu_)}(vov_unified, vov_per_gpu_)
end

const MultiGPUNeighborhoodSearch = GridNeighborhoodSearch{<:Any, <:Any, <:FullGridCellList{<:MultiGPUVectorOfVectors}}

KernelAbstractions.@kernel function kernel3(vov, vov_unified, previous_lengths, ::Val{gpu}, ::Val{islast}) where {gpu, islast}
    cell = KernelAbstractions.@index(Global)

    length = @inbounds vov.lengths[cell]
    if length > 0 || islast
        offset = 0
        for length_ in previous_lengths
            offset += @inbounds length_[cell]
        end

        for i in 1:length
            @inbounds vov_unified.backend[offset + i, cell] = vov.backend[i, cell]
        end

        if islast
            @inbounds vov_unified.lengths[cell] = offset + length
        end
    end
end

function PointNeighbors.initialize_grid!(neighborhood_search::MultiGPUNeighborhoodSearch, y::AbstractMatrix)
    (; cell_list) = neighborhood_search
    (; cells) = cell_list
    (; vov_unified, vov_per_gpu) = cells

    for vov_ in vov_per_gpu
        vov_.lengths .= 0
    end

    if neighborhood_search.search_radius < eps()
        # Cannot initialize with zero search radius.
        # This is used in TrixiParticles when a neighborhood search is not used.
        return neighborhood_search
    end

    # Fill cell lists per GPU
    parallel_foreach2(axes(y, 2), vov_per_gpu) do point, vov
        # Get cell index of the point's cell
        point_coords = extract_svector(y, Val(ndims(neighborhood_search)), point)
        cell = cell_coords(point_coords, neighborhood_search)

        # Add point to corresponding cell
        @boundscheck PointNeighbors.check_cell_bounds(cell_list, cell)

        # The atomic version of `pushat!` uses atomics to avoid race conditions when
        # `pushat!` is used in a parallel loop.
        @inbounds PointNeighbors.pushat_atomic!(vov, cell_index(cell_list, cell), point)
    end

    n_gpus = length(CUDA.devices())
    backend = CUDABackend()

    # Synchronize each device
    for i in 1:n_gpus
        CUDA.device!(i - 1)
        KernelAbstractions.synchronize(backend)
    end

    # Launch kernel on each device
    for gpu in 1:n_gpus
        # Select the correct device
        CUDA.device!(gpu - 1)

        previous_lengths = ntuple(gpu -> vov_per_gpu[gpu].lengths, gpu - 1)

        # Call the generic kernel, which only calls a function with the global GPU index
        kernel3(backend)(vov_per_gpu[gpu], vov_unified, previous_lengths, Val(gpu), Val(gpu == n_gpus), ndrange = length(vov_per_gpu[gpu]))
    end

    # Synchronize each device
    for i in 1:n_gpus
        CUDA.device!(i - 1)
        KernelAbstractions.synchronize(backend)
    end

    # Select first device again
    CUDA.device!(0)

    # # Merge cell lists
    # parallel_foreach2(axes(vov_unified, 2), y, partition = false) do cell, gpu
    #     offset = offsets[gpu][cell]

    #     points = vov_per_gpu[gpu][cell]
    #     for i in eachindex(points)
    #         vov_unified.backend[offset + i, cell] = points[i]
    #     end
    # end

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
