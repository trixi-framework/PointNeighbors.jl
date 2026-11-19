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

    new_length = @inbounds lengths[i] += 1

    # Inner bounds check. This is not skipped with `@inbounds`, as it is cheap
    # and writing out of bounds would silently corrupt the other vectors.
    check_list_bounds(vov, i, new_length)

    # Activate new entry in column `i`
    @inbounds backend[new_length, i] = value

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

    # Inner bounds check. This is not skipped with `@inbounds`, as it is cheap
    # and writing out of bounds would silently corrupt the other vectors.
    check_list_bounds(vov, i, new_length)

    # We can write here without race conditions, since the atomic add guarantees
    # that `new_length` is different for each thread.
    @inbounds backend[new_length, i] = value

    return vov
end

@inline function check_list_bounds(vov::DynamicVectorOfVectors, i, new_length)
    if new_length > size(vov.backend, 1)
        Atomix.@atomic vov.lengths[i] -= 1
        error("list is full. Use a larger `max_points_per_cell` or `max_neighbors`.")
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

# Data structure that behaves like a `Vector{Vector{T}}`. We call each inner vector a "bin".
# All bins are stored in one flat vector `permutation`: first all values of bin 1,
# then all values of bin 2, and so on.
# `first_bin_index[bin]` is the index in `permutation` where `bin` starts.
#
# The values are always the integers `1:n_values` (e.g., point indices),
# and each value is in at most one bin.
# Unlike `DynamicVectorOfVectors`, values cannot be added with `push!`.
# Instead, `update!` rebuilds the entire data structure from a function that
# assigns a bin to each value.
# Values without a bin (see `active_values` in `update!`) are stored after the last bin,
# so `permutation` is always a permutation of `1:n_values`.
#
# By default, `values` is the same array as `permutation`, so that each bin is a view
# of `permutation`.
# When the values are indices of points, reordering the points with `permutation`
# makes the points of each bin contiguous in memory.
# The values of the reordered points are then simply `1:n_values`, so that each bin
# is a range of indices.
# With `sorted = true`, `values` is `nothing` to represent this.
# The user is then responsible for reordering the points after every update.
struct CompactVectorOfVectors{T, V, P, I, B} <: AbstractVector{Vector{T}}
    values          :: V # either a reference to `permutation` or `nothing` (see above)
    permutation     :: P # Values `1:n_values`, sorted by bin
    first_bin_index :: I # Index of the first value of each bin, plus one entry (`n_values + 1`)
    bin_counts      :: I # Buffer for `update!`: number of values in each bin
    value_bins      :: B # Buffer for `update!`: bin of each value and index in this bin
    tail_index      :: I # Buffer for `update!`: index of the unassigned values after all bins

    # This constructor is necessary for Adapt.jl to work with this struct.
    # See the comments in gpu.jl for more details.
    function CompactVectorOfVectors(values, permutation, first_bin_index, bin_counts,
                                    value_bins, tail_index)
        new{eltype(permutation), typeof(values), typeof(permutation),
            typeof(first_bin_index), typeof(value_bins)}(values, permutation,
                                                         first_bin_index, bin_counts,
                                                         value_bins, tail_index)
    end
end

function CompactVectorOfVectors{T}(; n_bins, sorted = false) where {T}
    permutation = T[]
    values = sorted ? nothing : permutation
    first_bin_index = ones(Int32, n_bins + 1)
    bin_counts = zeros(Int32, n_bins)
    value_bins = NTuple{2, Int32}[]
    tail_index = Int32[]

    return CompactVectorOfVectors(values, permutation, first_bin_index, bin_counts,
                                  value_bins, tail_index)
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

# When the values are sorted (see above), each bin is simply a range of indices.
@inline bin_view(::Nothing, range) = range
@inline bin_view(values, range) = view(values, range)

# Sort all values in `1:n_values` by bin.
# The values that are not in `active_values` are not assigned to any bin
# and instead are stored in increasing order after all bins.
#
# This is a parallel counting sort, which also works on GPUs.
# Note that this is not a stable sort and the result is non-deterministic (in parallel).
#
# `active_values` must be unique and only contain values in `1:n_values`.
function update!(vov::CompactVectorOfVectors, value_to_bin, n_values,
                 parallelization_backend; active_values = Base.OneTo(n_values))
    (; permutation, first_bin_index, bin_counts, value_bins, tail_index) = vov

    eachvalue = Base.OneTo(n_values)

    # Resize data structures if necessary.
    if length(permutation) != n_values
        resize!(permutation, n_values)
        resize!(value_bins, n_values)
    end

    # We assume that `active_values` is unique and only contains values in `1:n_values`.
    all_values_active = length(active_values) == n_values

    if !all_values_active
        # Mark all values as unassigned (bin zero).
        @threaded parallelization_backend for value in eachvalue
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
    # Use a `view` to avoid scalar indexing on GPUs.
    fill!(view(first_bin_index, 1:1), 1)

    if !all_values_active
        # The `i`-th unassigned value is stored at index `i` after all bins.
        # Count the unassigned values with a cumulative sum, for which we can use
        # `permutation` as a temporary buffer, since it is overwritten below.
        if length(tail_index) != n_values
            resize!(tail_index, n_values)
        end

        @threaded parallelization_backend for value in eachvalue
            bin, _ = @inbounds value_bins[value]
            @inbounds permutation[value] = iszero(bin)
        end
        cumsum!(tail_index, permutation)
    end

    # Write each value to its index in its bin.
    n_bins = length(bin_counts)
    @threaded parallelization_backend for value in eachvalue
        bin, index_in_bin = @inbounds value_bins[value]

        if iszero(bin)
            index = @inbounds first_bin_index[n_bins + 1] + tail_index[value] - 1
        else
            index = @inbounds first_bin_index[bin] + index_in_bin - 1
        end
        @inbounds permutation[index] = value
    end

    return vov
end
