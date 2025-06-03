abstract type AbstractCellList end

# For the `DictionaryCellList`, this is a `KeySet`, which has to be `collect`ed first to be
# able to be used in a threaded loop.
@inline each_cell_index_threadable(cell_list::AbstractCellList) = each_cell_index(cell_list)

@inline function check_cell_bounds(cell_list::AbstractCellList, cell::Integer)
    (; cells) = cell_list

    checkbounds(cells, cell)
end

function construct_backend(_, ::Type{Vector{Vector{T}}},
                           max_outer_length,
                           max_inner_length) where {T}
    return [T[] for _ in 1:max_outer_length]
end

function construct_backend(_, ::Type{CompactVectorOfVectors{T}},
                           max_outer_length, _) where {T}
    return CompactVectorOfVectors{T}(n_bins = max_outer_length)
end

function construct_backend(_, ::Type{DynamicVectorOfVectors{T}},
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
function construct_backend(cell_list, ::Type{DynamicVectorOfVectors{T1, T2, T3, T4}},
                           max_outer_length,
                           max_inner_length) where {T1, T2, T3, T4}
    return construct_backend(cell_list, DynamicVectorOfVectors{T1}, max_outer_length,
                             max_inner_length)
end

function construct_backend(cell_list, ::Type{CompactVectorOfVectors{T1, T2, T3, T4}},
                           max_outer_length,
                           max_inner_length) where {T1, T2, T3, T4}
    return construct_backend(cell_list, CompactVectorOfVectors{T1}, max_outer_length,
                             max_inner_length)
end

function max_points_per_cell(cells::DynamicVectorOfVectors)
    return size(cells.backend, 1)
end

# Fallback when backend is a `Vector{Vector{T}}`. Only used for copying the cell list.
function max_points_per_cell(cells)
    return 100
end

include("dictionary.jl")
include("full_grid.jl")
include("spatial_hashing.jl")
