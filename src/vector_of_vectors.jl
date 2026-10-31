# Data structure that behaves like a `Vector{Vector}`, but uses a contiguous memory layout.
# Similar to `VectorOfVectors` of ArraysOfArrays.jl, but allows to resize the inner vectors.
struct DynamicVectorOfVectors{T, ARRAY2D, ARRAY1D, L} <: AbstractVector{Array{T, 1}}
    backend::ARRAY2D # Array{T, 2}, where each column represents a vector
    length_::L # Ref{Int32}: Number of vectors
    lengths::ARRAY1D # Array{Int32, 1} storing the lengths of the vectors

    # This constructor is necessary for Adapt.jl to work with this struct.
    # See the comments in gpu.jl for more details.
    function DynamicVectorOfVectors(backend, length_, lengths)
        new{eltype(backend), typeof(backend),
            typeof(lengths), typeof(length_)}(backend, length_, lengths)
    end
end

function DynamicVectorOfVectors{T}(; max_outer_length, max_inner_length,
                                   transpose_backend = false) where {T}
    if transpose_backend
        # Create a row-major array
        backend_ = Array{T, 2}(undef, max_outer_length, max_inner_length)
        # Wrap in a `PermutedDimsArray` to obtain the usual column-major access,
        # even though the underlying memory layout is row-major.
        backend = PermutedDimsArray(backend_, (2, 1))
    else
        backend = Array{T, 2}(undef, max_inner_length, max_outer_length)
    end
    length_ = Ref(zero(Int32))
    lengths = zeros(Int32, max_outer_length)

    return DynamicVectorOfVectors(backend, length_, lengths)
end

@inline Base.size(vov::DynamicVectorOfVectors) = (vov.length_[],)

@inline default_backend(x::DynamicVectorOfVectors) = default_backend(x.backend)

@inline function Base.getindex(vov::DynamicVectorOfVectors, i)
    (; backend, lengths) = vov

    # This is slightly faster than without explicit boundscheck and `@inbounds` below
    @boundscheck checkbounds(vov, i)

    return @inbounds view(backend, 1:lengths[i], i)
end

@inline function Base.push!(vov::DynamicVectorOfVectors, vector::AbstractVector)
    (; backend, length_, lengths) = vov

    # This data structure only supports one-based indexing
    Base.require_one_based_indexing(vector)

    # Activate a new column of `backend`
    j = length_[] += 1
    lengths[j] = length(vector)

    # Fill the new column
    for i in eachindex(vector)
        backend[i, j] = vector[i]
    end

    return vov
end

@inline function Base.push!(vov::DynamicVectorOfVectors, vector::AbstractVector, vectors...)
    push!(vov, vector)
    push!(vov, vectors...)
end

# `push!(vov[i], value)`
@inline function pushat!(vov::DynamicVectorOfVectors, i, value)
    (; backend, lengths) = vov

    # Outer bounds check
    @boundscheck checkbounds(vov, i)

    @inbounds lengths[i] += 1

    # Inner bounds check
    @boundscheck check_list_bounds(vov, i)

    # Activate new entry in column `i`
    @inbounds backend[lengths[i], i] = value

    return vov
end

@inline function pushat!(vov::Vector{<:Vector{<:Any}}, i, value)
    push!(vov[i], value)

    return vov
end

@inline function pushat_atomic!(vov::DynamicVectorOfVectors, i, value)
    (; backend, lengths) = vov

    # Outer bounds check
    @boundscheck checkbounds(vov, i)

    # Increment the column length with an atomic add to avoid race conditions.
    # Store the new value since it might be changed immediately afterwards by another
    # thread.
    new_length = @inbounds Atomix.@atomic lengths[i] += 1

    # Inner bounds check
    @boundscheck check_list_bounds(vov, i)

    # We can write here without race conditions, since the atomic add guarantees
    # that `new_length` is different for each thread.
    @inbounds backend[new_length, i] = value

    return vov
end

@inline function check_list_bounds(vov::DynamicVectorOfVectors, i)
    (; backend, lengths) = vov

    if lengths[i] > size(backend, 1)
        Atomix.@atomic lengths[i] -= 1
        error("cell list is full. Use a larger `max_points_per_cell`.")
    end
end

# `deleteat!(vov[i], j)`
@inline function deleteatat!(vov::DynamicVectorOfVectors, i, j)
    (; backend, lengths) = vov

    # Outer bounds check
    @boundscheck checkbounds(vov, i)
    # Inner bounds check
    @boundscheck checkbounds(1:lengths[i], j)

    # Replace value to delete by the last value in this column
    last_value = backend[lengths[i], i]
    @inbounds backend[j, i] = last_value

    # Remove the last value in this column
    @inbounds lengths[i] -= 1

    return vov
end

@inline function deleteatat!(vov::Vector{<:Vector{<:Any}}, i, j)
    deleteat!(vov[i], j)

    return vov
end

@inline function Base.empty!(vov::DynamicVectorOfVectors)
    # Move all pointers to the beginning
    vov.lengths .= zero(Int32)
    vov.length_[] = zero(Int32)

    return vov
end

# `empty!(vov[i])`
@inline function emptyat!(vov::DynamicVectorOfVectors, i)
    # Move length pointer to the beginning
    vov.lengths[i] = zero(Int32)

    return vov
end

@inline function emptyat!(vov::Vector{<:Vector{<:Any}}, i)
    Base.empty!(vov[i])
end

@inline function Base.resize!(vov::DynamicVectorOfVectors, n)
    # Make sure that all newly added vectors are empty
    vov.lengths[(length(vov) + 1):n] .= zero(Int32)
    vov.length_[] = n

    return vov
end

# Sort each inner vector
@inline function sorteach!(vov::DynamicVectorOfVectors)
    @threaded default_backend(vov.backend) for i in eachindex(vov)
        # QuickSort is ~1.6x faster here than the default
        sort!(view(vov.backend, 1:vov.lengths[i], i), alg = QuickSort)
    end

    return vov
end

# GPU version
@inline function sorteach!(vov::DynamicVectorOfVectors{<:Any, <:Any, <:AbstractGPUArray})
    # TODO remove this check when Metal supports sorting
    if nameof(typeof(default_backend(vov.backend))) == :MetalBackend
        @warn "sorting neighbor lists is not supported on Metal. Skipping sort."
        return vov
    end

    # Note that we cannot just do `sort!(vov[i])` on GPUs because that would call `sort!`
    # from within a GPU kernel, but this function is not GPU-compatible.
    # We might be able to use a sorting function from AcceleratedKernels.jl,
    # but for now the following workaround should be sufficient.

    # Set all unused entries to `typemax` so that they are sorted to the end
    @threaded default_backend(vov.backend) for i in axes(vov.backend, 2)
        for j in (vov.lengths[i] + 1):size(vov.backend, 1)
            @inbounds vov.backend[j, i] = typemax(eltype(vov.backend))
        end
    end

    # Now we can sort full columns.
    # Note that this will forward to highly optimized sorting functions on GPUs.
    # It currently does not work on Metal.
    sort!(vov.backend, dims = 1)

    return vov
end

@inline function sorteach!(vov::Vector{<:Vector{T}}) where {T}
    sort!.(vov)

    return vov
end

function max_inner_length(cells::DynamicVectorOfVectors, fallback)
    return size(cells.backend, 1)
end

# Fallback when backend is a `Vector{Vector{T}}`. Only used for copying the cell list.
@inline function max_inner_length(::Any, fallback)
    return fallback
end

@inline function transposed_backend(::DynamicVectorOfVectors{<:Any, <:PermutedDimsArray})
    return true
end

@inline function transposed_backend(::Any)
    return false
end

# Data structure that behaves like a `Vector{Vector}`, but stores all values contiguously,
# sorted by the inner vector (the "bin") they belong to.
# As opposed to `DynamicVectorOfVectors`, this data structure does not support
# modifying operations like `push!`.
# Instead, it is updated with `update!` by assigning a bin to each value in `1:n_values`.
# After the update, `values` is a permutation of `1:n_values`, sorting the values by bin.
# Values that are not assigned to any bin (see `update!`) are stored after all bins.
#
# When the values are indices of points, reordering the points with this permutation
# makes the points of each bin contiguous in memory.
# The values of the reordered points are then simply `1:n_values`,
# so that `values` can be replaced by `Base.OneTo(n_values)`.
struct CompactVectorOfVectors{T, V, I, B} <: AbstractVector{Vector{T}}
    values          :: V # Values `1:n_values`, sorted by bin
    first_bin_index :: I # Index in `values` of the first value of each bin, plus one entry
    bin_counts      :: I # Buffer for `update!`: number of values in each bin
    value_bins      :: B # Buffer for `update!`: bin of each value and index in this bin
    tail_index      :: I # Buffer for `update!`: index of the unassigned values after all bins

    # This constructor is necessary for Adapt.jl to work with this struct.
    # See the comments in gpu.jl for more details.
    function CompactVectorOfVectors(values, first_bin_index, bin_counts, value_bins,
                                    tail_index)
        new{eltype(values), typeof(values), typeof(first_bin_index),
            typeof(value_bins)}(values, first_bin_index, bin_counts, value_bins,
                                tail_index)
    end
end

function CompactVectorOfVectors{T}(; n_bins) where {T}
    values = T[]
    first_bin_index = ones(Int32, n_bins + 1)
    bin_counts = zeros(Int32, n_bins)
    value_bins = NTuple{2, Int32}[]
    tail_index = Int32[]

    return CompactVectorOfVectors(values, first_bin_index, bin_counts, value_bins,
                                  tail_index)
end

@inline Base.size(vov::CompactVectorOfVectors) = (length(vov.bin_counts),)

@inline default_backend(vov::CompactVectorOfVectors) = default_backend(vov.first_bin_index)

@inline function Base.getindex(vov::CompactVectorOfVectors, i)
    (; values, first_bin_index) = vov

    @boundscheck checkbounds(vov, i)

    first_index = @inbounds first_bin_index[i]
    last_index = @inbounds first_bin_index[i + 1] - 1

    return bin_view(values, first_index:last_index)
end

# When the values are the trivial permutation `1:n_values`, each bin is simply a range
@inline bin_view(values::AbstractUnitRange, range) = range
@inline bin_view(values, range) = view(values, range)

# Assign each value in `active_values` to the bin `value_to_bin(value)`
# and sort all values in `1:n_values` by bin.
# The remaining values that are not in `active_values` are not assigned to any bin.
# They are stored in increasing order after all bins.
# The values in each bin are sorted in increasing order as well,
# so that the result is deterministic.
#
# This is a parallel counting sort, which also works on GPUs.
function update!(vov::CompactVectorOfVectors, value_to_bin::F, n_values,
                 parallelization_backend; active_values = Base.OneTo(n_values)) where {F}
    (; values, first_bin_index, bin_counts, value_bins, tail_index) = vov

    if length(values) != n_values
        resize!(values, n_values)
        resize!(value_bins, n_values)
    end

    all_values_active = active_values isa AbstractUnitRange &&
                        active_values == Base.OneTo(n_values)

    if !all_values_active
        # Mark all values as unassigned (bin zero)
        @threaded parallelization_backend for value in Base.OneTo(n_values)
            @inbounds value_bins[value] = (zero(Int32), zero(Int32))
        end
    end

    # Count the values in each bin. The atomic increment also yields the index of
    # this value in its bin, so that only one atomic operation per value is needed.
    fill!(bin_counts, 0)
    @threaded parallelization_backend for value in active_values
        bin = value_to_bin(value)
        index_in_bin = @inbounds Atomix.@atomic bin_counts[bin] += one(Int32)
        @inbounds value_bins[value] = (Int32(bin), index_in_bin)
    end

    # `first_bin_index[bin] = 1 + sum(bin_counts[1:(bin - 1)])`
    first_bin_index_tail = view(first_bin_index, 2:length(first_bin_index))
    cumsum!(first_bin_index_tail, bin_counts)
    first_bin_index_tail .+= 1
    fill!(view(first_bin_index, 1:1), 1)

    if !all_values_active
        # The `i`-th unassigned value is stored at index `i` after all bins.
        # Count the unassigned values with a cumulative sum, for which we can use `values`
        # as a temporary buffer, since it is overwritten below.
        if length(tail_index) != n_values
            resize!(tail_index, n_values)
        end

        @threaded parallelization_backend for value in Base.OneTo(n_values)
            bin, _ = @inbounds value_bins[value]
            @inbounds values[value] = iszero(bin)
        end
        cumsum!(tail_index, values)
    end

    # Write each value to its index in its bin
    n_bins = length(bin_counts)
    @threaded parallelization_backend for value in Base.OneTo(n_values)
        bin, index_in_bin = @inbounds value_bins[value]

        if iszero(bin)
            index = @inbounds first_bin_index[n_bins + 1] + tail_index[value] - 1
        else
            index = @inbounds first_bin_index[bin] + index_in_bin - 1
        end
        @inbounds values[index] = value
    end

    # The order inside each bin depends on the order of the atomic operations above.
    # Sort the values inside each bin to make the result deterministic.
    # Bins are usually small, so we use a simple insertion sort.
    @threaded parallelization_backend for bin in Base.OneTo(n_bins)
        first_index = @inbounds first_bin_index[bin]
        last_index = @inbounds first_bin_index[bin + 1] - 1

        for i in (first_index + 1):last_index
            value = @inbounds values[i]
            j = i - 1
            while j >= first_index && @inbounds(values[j]) > value
                @inbounds values[j + 1] = values[j]
                j -= 1
            end
            @inbounds values[j + 1] = value
        end
    end

    return vov
end
