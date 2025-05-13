@inline function check_cell_bounds(cell_list::AbstractCellList, cell::Integer)
    (; cells) = cell_list

    checkbounds(cells, cell)
end

# This became kinda unnecessary now since we mostly (only) call the function with the hash key (Integer)
# # Move that to spatial_hashing.jl
@inline function check_cell_bounds(cell_list::SpatialHashingCellList, cell::Tuple)
    check_cell_bounds(cell_list, spatial_hash(cell, cell_list.list_size))
end

# We need the prod() because FullGridCellList's size is a tuple of cells per dimension whereas
# SpatialHashingCellList's size is an Integer for the number of cells in total.
function construct_backend(::Type{<:AbstractCellList}, ::Type{Vector{Vector{T}}}, size,
                           max_points_per_cell) where {T}
    return [T[] for _ in 1:prod(size)]
end

function construct_backend(::Type{<:AbstractCellList}, ::Type{DynamicVectorOfVectors{T}},
                           size,
                           max_points_per_cell) where {T}
    cells = DynamicVectorOfVectors{T}(max_outer_length = prod(size),
                                      max_inner_length = max_points_per_cell)
    resize!(cells, prod(size))

    return cells
end

# When `typeof(cell_list.cells)` is passed, we don't pass the type
# `DynamicVectorOfVectors{T}`, but a type `DynamicVectorOfVectors{T1, T2, T3, T4}`.
# While `A{T} <: A{T1, T2}`, this doesn't hold for the types.
# `Type{A{T}} <: Type{A{T1, T2}}` is NOT true.
function construct_backend(cell_list::Type{<:AbstractCellList},
                           ::Type{DynamicVectorOfVectors{T1, T2, T3, T4}}, size,
                           max_points_per_cell) where {T1, T2, T3, T4}
    return construct_backend(cell_list, DynamicVectorOfVectors{T1}, size,
                             max_points_per_cell)
end
