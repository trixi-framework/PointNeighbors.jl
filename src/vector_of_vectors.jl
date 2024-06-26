# Data structure that behaves like a `Vector{Vector}`, but uses a contiguous memory layout.
# Similar to `VectorOfVectors` of ArraysOfArrays.jl, but allows to resize the inner vectors.
struct DynamicVectorOfVectors{T, ARRAY2D, ARRAY1D} <: AbstractVector{Array{T, 1}}
    backend::ARRAY2D # Array{T, 2}, where each column represents a vector
    length_::Base.RefValue{Int32} # Number of vectors
    lengths::ARRAY1D # Array{Int32, 1} storing the lengths of the vectors
end

function DynamicVectorOfVectors{T}(; max_outer_length, max_inner_length) where {T}
    backend = Array{T, 2}(undef, max_inner_length, max_outer_length)
    length_ = Ref(zero(Int32))
    lengths = zeros(Int32, max_outer_length)

    return DynamicVectorOfVectors{T, typeof(backend), typeof(lengths)}(backend, length_,
                                                                       lengths)
end

@inline Base.size(vov::DynamicVectorOfVectors) = (vov.length_[],)

@inline function Base.getindex(vov::DynamicVectorOfVectors, i)
    (; backend, lengths) = vov

    @boundscheck checkbounds(vov, i)

    return view(backend, 1:lengths[i], i)
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

    @boundscheck checkbounds(vov, i)

    # Activate new entry in column `i`
    backend[lengths[i] += 1, i] = value

    return vov
end

@inline function pushat!(vov::Vector{<:Vector{<:Any}}, i, value)
    push!(vov[i], value)

    return vov
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
    backend[j, i] = last_value

    # Remove the last value in this column
    lengths[i] -= 1

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
