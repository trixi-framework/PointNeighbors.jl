struct DictionaryCellList{NDIMS}
    hashtable    :: Dict{NTuple{NDIMS, Int}, Vector{Int}}
    empty_vector :: Vector{Int} # Just an empty vector (used in `eachneighbor`)

    function DictionaryCellList{NDIMS}() where {NDIMS}
        hashtable = Dict{NTuple{NDIMS, Int}, Vector{Int}}()
        empty_vector = Int[]

        new{NDIMS}(hashtable, empty_vector)
    end
end

function Base.empty!(cell_list::DictionaryCellList)
    Base.empty!(cell_list.hashtable)

    return cell_list
end

function push_cell!(cell_list::DictionaryCellList, cell, particle)
    (; hashtable) = cell_list

    if haskey(hashtable, cell)
        append!(hashtable[cell], particle)
    else
        hashtable[cell] = [particle]
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

@inline eachcell(cell_list::DictionaryCellList) = keys(cell_list.hashtable)

@inline function Base.getindex(cell_list::DictionaryCellList, cell)
    (; hashtable, empty_vector) = cell_list

    # Return an empty vector when `cell_index` is not a key of `hashtable` and
    # reuse the empty vector to avoid allocations.
    return get(hashtable, cell, empty_vector)
end
