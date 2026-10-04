"""
!!! warning "Experimental Implementation"
    This is an experimental feature and may change in any future releases.
"""

"""
    TreeGridCellList(; min_corner, max_corner, backend = DynamicVectorOfVectors{Int32}, 
                   max_points_per_cell = 100, max_level = 8)

A grid-based cell list, similar to [`FullGridCellList`](@ref). Here, the grid resolution is determined by a hierarchical tree depth (`max_level`), compared to  [`FullGridCellList`](@ref) where the search radius is used.
The grid is divided into `2^max_level` cells along each dimension.
# Arguments
- `min_corner`: Coordinates of the domain corner in negative coordinate directions.
- `max_corner`: Coordinates of the domain corner in positive coordinate directions.
- `backend = DynamicVectorOfVectors{Int32}`: Type of the data structure to store the actual
    cell lists. Can be
    - `Vector{Vector{Int32}}`: Scattered memory, but very memory-efficient.
    - `DynamicVectorOfVectors{Int32}`: Contiguous memory, optimizing cache-hits
                                       and GPU-compatible.
- `max_points_per_cell = 100`: Maximum number of points per cell. This will be used to
                               allocate the `DynamicVectorOfVectors`. It is not used with
                               the `Vector{Vector{Int32}}` backend.
- `max_level::Int`: The subdivision level determining grid resolution. Total cells allocated will be `(2^max_level)^NDIMS`.
"""
struct TreeGridCellList{C, LI, MINC, MAXC, CL, ML, CPC} <: AbstractCellList
    cells          :: C
    linear_indices :: LI
    min_corner     :: MINC
    max_corner     :: MAXC

    cell_levels       :: CL
    max_level         :: ML
    capacity_per_cell :: CPC
end

@inline Base.ndims(cell_list::TreeGridCellList) = ndims(cell_list.linear_indices)

function supported_update_strategies(::TreeGridCellList)
    return (ParallelUpdate,)
end

function TreeGridCellList(; min_corner, max_corner,
                          backend = DynamicVectorOfVectors{Int32},
                          max_points_per_cell = 100, max_level = 8, capacity_per_cell = 1)
    if length(min_corner) != length(max_corner)
        throw(ArgumentError("min_corner and max_corner must have the same length"))
    end

    NDIMS = length(min_corner)
    n_cells_per_dimension = 2^max_level

    if NDIMS != 2 || NDIMS != 3
        throw(ArgumentError("TreeGridCellList only supports 2 or 3 dimensions."))
    end

    # Pad domain a little more to avoid 0 in cell indices due to rounding errors.
    length_grid = maximum(max_corner .- min_corner)
    length_cell = length_grid / (2^max_level)
    min_corner = SVector(Tuple(min_corner .- (1001 // 1000 * length_cell)))
    max_corner = SVector(Tuple(max_corner .+ (1001 // 1000 * length_cell)))

    if max_level < 0
        # Create an empty "template" cell list to be used with `copy_cell_list`
        cells = construct_backend(backend, 0, max_points_per_cell)
        linear_indices = LinearIndices(ntuple(_ -> 0, length(min_corner)))
        cell_levels = Vector{Int8}(undef, 0)
    else
        n_cells = n_cells_per_dimension^NDIMS
        linear_indices = LinearIndices(ntuple(_ -> n_cells_per_dimension, NDIMS))
        cells = construct_backend(backend, n_cells, max_points_per_cell)
        cell_levels = Vector{Int8}(undef, n_cells)
    end

    return TreeGridCellList(cells, linear_indices, min_corner, max_corner, cell_levels,
                            max_level, capacity_per_cell)
end

@inline function morton_cell_index(coords, cell_list::TreeGridCellList,
                                   level = cell_list.max_level)
    cartesian_coords = cartesian_cell_coords(coords, cell_list, level)
    return cartesian_to_morton(cartesian_coords)
end

@inline function cartesian_cell_coords(coords, cell_list::TreeGridCellList,
                                       level = cell_list.max_level)
    (; min_corner) = cell_list
    length_grid = grid_length(cell_list)
    cell_length = length_grid / (2^level)

    NDIMS = length(min_corner)

    return SVector{NDIMS, Int}(ntuple(Val(NDIMS)) do i
                                   @inbounds floor_to_int((coords[i] - min_corner[i]) /
                                                          cell_length) + 1
                               end)
end

function Base.empty!(cell_list::TreeGridCellList)
    (; cells, cell_levels) = cell_list
    cell_levels .= -1

    # `Base.empty!.(cells)`, but for all backends
    @threaded default_backend(cells) for i in eachindex(cells)
        emptyat!(cells, i)
    end

    return cell_list
end

function push_cell!(cell_list::TreeGridCellList, cell, particle)
    (; cells) = cell_list
    @inbounds pushat!(cells, cell_index(cell_list, cell), particle)

    return cell_list
end

@inline function push_cell_atomic!(cell_list::TreeGridCellList, cell, particle)
    (; cells) = cell_list

    # `push!(cell_list[cell], particle)`, but for all backends.
    # The atomic version of `pushat!` uses atomics to avoid race conditions when `pushat!`
    # is used in a parallel loop.
    @inbounds pushat_atomic!(cells, cell_index(cell_list, cell), particle)

    return cell_list
end

function deleteat_cell!(cell_list::TreeGridCellList, cell, i)
    (; cells) = cell_list

    # `deleteat!(cell_list[cell], i)`, but for all backends
    deleteatat!(cells, cell_index(cell_list, cell), i)
end

@inline each_cell_index(cell_list::TreeGridCellList) = eachindex(cell_list.cells)

@propagate_inbounds function cell_index(::TreeGridCellList, cell::Tuple)
    return cartesian_to_morton(cell...)
end

@inline cell_index(::TreeGridCellList, cell::Integer) = cell

@propagate_inbounds function Base.getindex(cell_list::TreeGridCellList, cell)
    (; cells) = cell_list

    return cells[cell_index(cell_list, cell)]
end

@inline function is_correct_cell(cell_list::TreeGridCellList, cell, cell_index_)
    return cell_index(cell_list, cell) == cell_index_
end

@inline index_type(::TreeGridCellList) = Int32

@inline is_leaf(cell_list, cell) = cell_list.cell_levels[cell_index(cell_list, cell)] > -1

function copy_cell_list(cell_list::TreeGridCellList)
    (; min_corner, max_corner, max_level, capacity_per_cell) = cell_list

    return TreeGridCellList(; min_corner, max_corner, max_level, capacity_per_cell,
                            backend = typeof(cell_list.cells),
                            max_points_per_cell = max_inner_length(cell_list.cells, 100))
end

@inline level_offset(cell_list::TreeGridCellList,
                     level) = (2^ndims(cell_list))^(cell_list.max_level - level)

@inline grid_length(cell_list::TreeGridCellList) = maximum(cell_list.max_corner .-
                                                           cell_list.min_corner)

@inline function cell_length(cell_list::TreeGridCellList)
    (; max_level) = cell_list

    return grid_length(cell_list) / (2^max_level)
end

@inline function morton_to_cartesian(::Val{2}, m::Integer)
    m_zero = m - 1
    return SVector{2, Int}(_Compact1By1(m_zero >> 0),
                           _Compact1By1(m_zero >> 1))
end

@inline function morton_to_cartesian(::Val{3}, m::Integer)
    m_zero = m - 1
    return SVector{3, Int}(_Compact1By2(m_zero >> 0),
                           _Compact1By2(m_zero >> 1),
                           _Compact1By2(m_zero >> 2))
end

@inline cartesian_to_morton(c::SVector{2, <:Integer}) = cartesian2morton(c)
@inline cartesian_to_morton(c::SVector{3, <:Integer}) = cartesian3morton(c)

@inline cartesian_to_morton(c::NTuple{2, <:Integer}) = cartesian2morton(SVector(c))
@inline cartesian_to_morton(c::NTuple{3, <:Integer}) = cartesian3morton(SVector(c))
