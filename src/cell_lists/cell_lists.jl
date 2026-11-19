abstract type AbstractCellList end

# Fallback is `false` for cell lists, so that new cell lists are conservative.
supports_infinite_domain(::AbstractCellList) = false

# For the `DictionaryCellList`, this is a `KeySet`, which has to be `collect`ed first to be
# able to be used in a threaded loop.
@inline each_cell_index_threadable(cell_list::AbstractCellList) = each_cell_index(cell_list)

@inline function check_cell_bounds(cell_list::AbstractCellList, cell::Integer)
    (; cells) = cell_list

    checkbounds(cells, cell)
end

function construct_backend(::Type{Vector{Vector{T}}},
                           max_outer_length, max_inner_length;
                           transpose_backend = false) where {T}
    if transpose_backend
        error("transpose backend is only supported for DynamicVectorOfVectors backend")
    end

    return [T[] for _ in 1:max_outer_length]
end

# Note that `Type{DynamicVectorOfVectors{T}} <: Type{<:DynamicVectorOfVectors{T}}`
# and `Type{DynamicVectorOfVectors{T1, T2, ...}} <: Type{<:DynamicVectorOfVectors{T1}}`.
# The latter is passed when `typeof(cell_list.cells)` is used to construct a new backend.
function construct_backend(::Type{<:DynamicVectorOfVectors{T}},
                           max_outer_length, max_inner_length;
                           transpose_backend = false) where {T}
    cells = DynamicVectorOfVectors{T}(; max_outer_length, max_inner_length,
                                      transpose_backend)
    resize!(cells, max_outer_length)

    return cells
end

# See the comment above for why `Type{<:CompactVectorOfVectors{T}}` is used.
function construct_backend(::Type{<:CompactVectorOfVectors{T}},
                           max_outer_length, max_inner_length;
                           transpose_backend = false) where {T}
    if transpose_backend
        error("transpose backend is only supported for DynamicVectorOfVectors backend")
    end

    return CompactVectorOfVectors{T}(; n_bins = max_outer_length)
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
include("tree.jl")
