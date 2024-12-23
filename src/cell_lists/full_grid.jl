"""
    FullGridCellList(; min_corner, max_corner, search_radius = 0.0,
                     periodicity = false, backend = DynamicVectorOfVectors{Int32},
                     max_points_per_cell = 100)

A simple cell list implementation where each (empty or non-empty) cell of a rectangular
(axis-aligned) domain is assigned a list of points.
This cell list only works when all points are inside the specified domain at all times.

Only set `min_corner` and `max_corner` and use the default values for the other arguments
to create an empty "template" cell list that can be used to create an empty "template"
neighborhood search.
See [`copy_neighborhood_search`](@ref) for more details.

# Keywords
- `min_corner`: Coordinates of the domain corner in negative coordinate directions.
- `max_corner`: Coordinates of the domain corner in positive coordinate directions.
- `search_radius = 0.0`: Search radius of the neighborhood search, which will determine the
                         cell size. Use the default of `0.0` to create a template (see above).
- `periodicity = false`: Set to `true` when using a [`PeriodicBox`](@ref) with the
                         neighborhood search. When using [`copy_neighborhood_search`](@ref),
                         this option can be ignored an will be set automatically depending
                         on the periodicity of the neighborhood search.
- `backend = DynamicVectorOfVectors{Int32}`: Type of the data structure to store the actual
    cell lists. Can be
    - `Vector{Vector{Int32}}`: Scattered memory, but very memory-efficient.
    - `DynamicVectorOfVectors{Int32}`: Contiguous memory, optimizing cache-hits.
- `max_points_per_cell = 100`: Maximum number of points per cell. This will be used to
                               allocate the `DynamicVectorOfVectors`. It is not used with
                               the `Vector{Vector{Int32}}` backend.
"""
struct FullGridCellList{C, LI, MINC, MAXC} <: AbstractCellList
    cells          :: C
    linear_indices :: LI
    min_corner     :: MINC
    max_corner     :: MAXC
end

function supported_update_strategies(::FullGridCellList{<:DynamicVectorOfVectors})
    return (ParallelUpdate, SemiParallelUpdate, SerialUpdate)
end

supported_update_strategies(::FullGridCellList) = (SemiParallelUpdate, SerialUpdate)

function FullGridCellList(; min_corner, max_corner, search_radius = 0.0,
                          backend = DynamicVectorOfVectors{Int32},
                          max_points_per_cell = 100)
    # Add one layer in each direction to make sure neighbor cells exist.
    # Also pad domain a little more to avoid 0 in cell indices due to rounding errors.
    # We can't just use `eps()`, as one might use lower precision types.
    # This padding is safe, and will give us one more layer of cells in the worst case.
    min_corner = SVector(Tuple(min_corner .- 1.001f0 * search_radius))
    max_corner = SVector(Tuple(max_corner .+ 1.001f0 * search_radius))

    if search_radius < eps()
        # Create an empty "template" cell list to be used with `copy_cell_list`
        cells = construct_backend(backend, 0, max_points_per_cell)
        linear_indices = nothing
    else
        # Note that we don't shift everything so that the first cell starts at `min_corner`.
        # The first cell is the cell containing `min_corner`, so we need to add one layer
        # in order for `max_corner` to be inside a cell.
        n_cells_per_dimension = ceil.(Int, (max_corner .- min_corner) ./ search_radius) .+ 1
        linear_indices = LinearIndices(Tuple(n_cells_per_dimension))

        cells = construct_backend(backend, n_cells_per_dimension, max_points_per_cell)
    end

    return FullGridCellList(cells, linear_indices, min_corner, max_corner)
end

function construct_backend(::Type{Vector{Vector{T}}}, size, max_points_per_cell) where {T}
    return [T[] for _ in 1:prod(size)]
end

function construct_backend(::Type{DynamicVectorOfVectors{T}}, size,
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
function construct_backend(::Type{DynamicVectorOfVectors{T1, T2, T3, T4}}, size,
                           max_points_per_cell) where {T1, T2, T3, T4}
    return construct_backend(DynamicVectorOfVectors{T1}, size, max_points_per_cell)
end

@inline function cell_coords(coords, periodic_box::Nothing, cell_list::FullGridCellList,
                             cell_size)
    (; min_corner) = cell_list

    # Subtract `min_corner` to offset coordinates so that the min corner of the grid
    # corresponds to the (1, 1, 1) cell.
    # Note that we use `min_corner == periodic_box.min_corner`, so we don't have to handle
    # periodic boxes differently, as they also use 1-based indexing.
    return Tuple(floor_to_int.((coords .- min_corner) ./ cell_size)) .+ 1
end

function Base.empty!(cell_list::FullGridCellList)
    (; cells) = cell_list

    # `Base.empty!.(cells)`, but for all backends
    for i in eachindex(cells)
        emptyat!(cells, i)
    end

    return cell_list
end

function Base.empty!(cell_list::FullGridCellList{Nothing})
    # This is an empty "template" cell list to be used with `copy_cell_list`
    error("`search_radius` is not defined for this cell list")
end

function push_cell!(cell_list::FullGridCellList, cell, particle)
    (; cells) = cell_list

    # `push!(cell_list[cell], particle)`, but for all backends
    pushat!(cells, cell_index(cell_list, cell), particle)

    return cell_list
end

function push_cell!(cell_list::FullGridCellList{Nothing}, cell, particle)
    # This is an empty "template" cell list to be used with `copy_cell_list`
    error("`search_radius` is not defined for this cell list")
end

@inline function push_cell_atomic!(cell_list::FullGridCellList, cell, particle)
    (; cells) = cell_list

    # `push!(cell_list[cell], particle)`, but for all backends.
    # The atomic version of `pushat!` uses atomics to avoid race conditions when `pushat!`
    # is used in a parallel loop.
    pushat_atomic!(cells, cell_index(cell_list, cell), particle)

    return cell_list
end

function deleteat_cell!(cell_list::FullGridCellList, cell, i)
    (; cells) = cell_list

    # `deleteat!(cell_list[cell], i)`, but for all backends
    deleteatat!(cells, cell_index(cell_list, cell), i)
end

@inline each_cell_index(cell_list::FullGridCellList) = eachindex(cell_list.cells)

function each_cell_index(cell_list::FullGridCellList{Nothing})
    # This is an empty "template" cell list to be used with `copy_cell_list`
    error("`search_radius` is not defined for this cell list")
end

@propagate_inbounds function cell_index(cell_list::FullGridCellList, cell::Tuple)
    (; linear_indices) = cell_list

    return linear_indices[cell...]
end

@inline cell_index(::FullGridCellList, cell::Integer) = cell

@propagate_inbounds function Base.getindex(cell_list::FullGridCellList, cell)
    (; cells) = cell_list

    return cells[cell_index(cell_list, cell)]
end

@inline function is_correct_cell(cell_list::FullGridCellList, cell_coords, cell_index_)
    return cell_index(cell_list, cell_coords) == cell_index_
end

@inline index_type(::FullGridCellList) = Int32

function copy_cell_list(cell_list::FullGridCellList, search_radius, periodic_box)
    (; min_corner, max_corner) = cell_list

    return FullGridCellList(; min_corner, max_corner, search_radius,
                            backend = typeof(cell_list.cells),
                            max_points_per_cell = max_points_per_cell(cell_list.cells))
end

function max_points_per_cell(cells::DynamicVectorOfVectors)
    return size(cells.backend, 1)
end

# Fallback when backend is a `Vector{Vector{T}}`. Only used for copying the cell list.
function max_points_per_cell(cells)
    return 100
end

function check_domain_bounds(cell_list::FullGridCellList, y, search_radius)
    if any(isnan, y)
        error("particle coordinates contain NaNs")
    end

    # Require one extra layer in each direction to make sure neighbor cells exist.
    # Convert to CPU in case this lives on the GPU.
    min_corner = Array(minimum(y, dims = 2) .- search_radius)
    max_corner = Array(maximum(y, dims = 2) .+ search_radius)

    if any(min_corner .< cell_list.min_corner) || any(max_corner .> cell_list.max_corner)
        error("particle coordinates are outside the domain bounds of the cell list")
    end
end
