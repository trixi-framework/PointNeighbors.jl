abstract type AbstractCellList end

# For the `DictionaryCellList`, this is a `KeySet`, which has to be `collect`ed first to be
# able to be used in a threaded loop.
@inline each_cell_index_threadable(cell_list::AbstractCellList) = each_cell_index(cell_list)

@inline function check_cell_bounds(cell_list::AbstractCellList, cell::Integer)
    (; cells) = cell_list

    checkbounds(cells, cell)
end

# We need the prod() because FullGridCellList's size is a tuple of cells per dimension whereas
# SpatialHashingCellList's size is an Integer for the number of cells in total.
function construct_backend(::Type{<:AbstractCellList}, ::Type{Vector{Vector{T}}},
                           max_outer_length,
                           max_inner_length) where {T}
    return [T[] for _ in 1:max_outer_length]
end

function construct_backend(::Type{<:AbstractCellList}, ::Type{DynamicVectorOfVectors{T}},
                           max_outer_length,
                           max_inner_length) where {T}
    cells = DynamicVectorOfVectors{T}(max_outer_length = max_outer_length,
                                      max_inner_length = max_inner_length)
    resize!(cells, max_outer_length)

    return cells
end

# When `typeof(cell_list.cells)` is passed, we don't pass the type
# `DynamicVectorOfVectors{T}`, but a type `DynamicVectorOfVectors{T1, T2, T3, T4}`.
# While `A{T} <: A{T1, T2}`, this doesn't hold for the types.
# `Type{A{T}} <: Type{A{T1, T2}}` is NOT true.
function construct_backend(cell_list::Type{<:AbstractCellList},
                           ::Type{DynamicVectorOfVectors{T1, T2, T3, T4}}, max_outer_length,
                           max_inner_length) where {T1, T2, T3, T4}
    return construct_backend(cell_list, DynamicVectorOfVectors{T1}, max_outer_length,
                             max_inner_length)
end

include("dictionary.jl")
include("full_grid.jl")
include("spatial_hashing.jl")
include("cell_lists_util.jl")
