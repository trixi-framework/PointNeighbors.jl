"""
    DictionaryCellList{T, NDIMS}()

A simple cell list implementation where a cell index `(i, j)` or `(i, j, k)` is mapped to a
`Vector{T}` by a `Dict`.
By using a dictionary, which only stores non-empty cells, the domain is
potentially infinite.

This implementation is very simple, but it neither uses an optimized hash function
for integer tuples, nor does it use a contiguous memory layout.
Consequently, this cell list is not GPU-compatible.

# Arguments
- `T`: Either an `Integer` type, or `PointWithCoordinates{NDIMS, ELTYPE}`.
       See [`PointWithCoordinates`](@ref) for more details.
- `NDIMS`: Number of dimensions.
"""
struct DictionaryCellList{T, NDIMS} <: AbstractCellList{T}
    hashtable    :: Dict{NTuple{NDIMS, Int}, Vector{T}}
    empty_vector :: Vector{T} # Just an empty vector (used in `getindex`)

    function DictionaryCellList{T, NDIMS}() where {T, NDIMS}
        hashtable = Dict{NTuple{NDIMS, Int}, Vector{T}}()
        empty_vector = T[]

        new{T, NDIMS}(hashtable, empty_vector)
    end
end

supported_update_strategies(::DictionaryCellList) = (SemiParallelUpdate, SerialUpdate)

function Base.empty!(cell_list::DictionaryCellList)
    Base.empty!(cell_list.hashtable)

    return cell_list
end

function push_cell!(cell_list::DictionaryCellList, cell, point)
    (; hashtable) = cell_list

    if haskey(hashtable, cell)
        append!(hashtable[cell], point)
    else
        hashtable[cell] = [point]
    end

    return cell_list
end

function deleteat_cell!(cell_list::DictionaryCellList, cell, i)
    (; hashtable) = cell_list

    # This works for `i::Integer`, `i::AbstractVector`, and even `i::Base.Generator`
    if length(hashtable[cell]) <= count(_ -> true, i)
        delete_cell!(cell_list, cell)
    else
        deleteat!(hashtable[cell], i)
    end
end

function delete_cell!(cell_list, cell)
    delete!(cell_list.hashtable, cell)
end

@inline each_cell_index(cell_list::DictionaryCellList) = keys(cell_list.hashtable)

# For this cell list, this is a `KeySet`, which has to be `collect`ed first to be
# able to be used in a threaded loop.
@inline function each_cell_index_threadable(cell_list::DictionaryCellList)
    return collect(each_cell_index(cell_list))
end

@inline function Base.getindex(cell_list::DictionaryCellList, cell)
    (; hashtable, empty_vector) = cell_list

    # Return an empty vector when `cell_index` is not a key of `hashtable` and
    # reuse the empty vector to avoid allocations.
    return get(hashtable, cell, empty_vector)
end

@inline function is_correct_cell(::DictionaryCellList, cell_coords, cell_index)
    return cell_coords == cell_index
end

@inline index_type(::DictionaryCellList{<:Any, NDIMS}) where {NDIMS} = NTuple{NDIMS, Int}

function copy_cell_list(::DictionaryCellList{T, NDIMS}, search_radius,
                        periodic_box) where {T, NDIMS}
    return DictionaryCellList{T, NDIMS}()
end
