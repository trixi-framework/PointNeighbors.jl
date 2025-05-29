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

function DynamicVectorOfVectors{T}(; max_outer_length, max_inner_length) where {T}
    backend = Array{T, 2}(undef, max_inner_length, max_outer_length)
    length_ = Ref(zero(Int32))
    lengths = zeros(Int32, max_outer_length)

    return DynamicVectorOfVectors(backend, length_, lengths)
end

@inline Base.size(vov::DynamicVectorOfVectors) = (vov.length_[],)

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

    lengths[i] += 1

    # Inner bounds check
    @boundscheck check_list_bounds(vov, i)

    # Activate new entry in column `i`
    backend[lengths[i], i] = value

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

# As opposed to `DynamicVectorOfVectors`, this data structure does not support
# modifying operations like `push!`.
# It can be updated by assigning each contained integer value a new bin index.
mutable struct StaticVectorOfVectors{T, V, L, I}
    backend         :: V # Vector{Int} containing all values sorted by bin
    n_bins          :: L # Ref{Int32}: Number of bins
    first_bin_index :: I # Vector{Int} containing the first index in `values` of each bin

    # This constructor is necessary for Adapt.jl to work with this struct.
    # See the comments in gpu.jl for more details.
    function StaticVectorOfVectors(backend::V, n_bins::L, first_bin_index::I) where {T, V, L, I}
        new{T,V,L,I}(backend, n_bins, first_bin_index)
    end
end

# Outer constructor that only takes n_values and n_bins, leaves backend = nothing
function StaticVectorOfVectors{T}(; n_bins::Int) where {T}
    fbi = Vector{Int}(undef, n_bins)
    fbi[1] = 1 # required for the update
    backend = nothing
    n_bins = Ref{Int32}(n_bins)

    return StaticVectorOfVectors{T, typeof(backend), typeof(n_bins), typeof(first_bin_index)}(
        backend,                # backend
        n_bins,                 # n_bins
        fbi                     # first_bin_index
    )
end

@inline Base.size(vov::StaticVectorOfVectors) = (vov.n_bins[],)

@inline function Base.getindex(vov::StaticVectorOfVectors, i)
    (; backend, first_bin_index) = vov
    
    start = first_bin_index[i]

    if i == length(first_bin_index)
        return view(backend, start:length(backend))
    end

    stop = first_bin_index[i + 1] - 1

    return view(backend, start:stop)
end

# Used to initialize the data 
@inline function initialize!(vov::StaticVectorOfVectors, values)
    vov.backend = values
    return vov
end

@inline function update!(vov::StaticVectorOfVectors, f)
    (; backend, first_bin_index, n_bins) = vov

    # TODO figure out how to do that fast and on the GPU
    sort!(backend, by = f)

    # TODO figure out how to do that fast and on the GPU
    n_particles_per_cell = [count(x -> f(x) == j, backend) for j in 1:n_bins[]]
    # Add 1 since first_bin_index starts at 1
    n_particles_per_cell[1] += 1

    # TODO avoid allocations
    first_bin_index[2:(end - 1)] .= cumsum(n_particles_per_cell)[1:(end - 1)]
end
