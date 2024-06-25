"""
    FullGridCellList(; min_corner, max_corner, search_radius = 0.0, periodicity = false)

A simple cell list implementation where each (empty or non-empty) cell of a rectangular
domain is assigned a list of points.
This cell list only works when all points are inside the specified domain at all times.

Use the default arguments to create an empty "template" cell list that can be used to create
an empty "template" neighborhood search.
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
"""
struct FullGridCellList{C, LI, MC}
    cells          :: C
    linear_indices :: LI
    min_cell       :: MC

    function FullGridCellList{C, LI, MC}(cells, linear_indices, min_cell) where {C, LI, MC}
        new{C, LI, MC}(cells, linear_indices, min_cell)
    end
end

function FullGridCellList(; min_corner, max_corner, search_radius = 0.0,
                          periodicity = false)
    if search_radius < eps()
        # Create an empty "template" cell list to be used with `copy_cell_list`
        cells = nothing
        linear_indices = nothing

        # Misuse `min_cell` to store min and max corner for copying
        min_cell = (min_corner, max_corner)
    else
        if periodicity
            # Subtract `min_corner` because that's how the grid NHS works with periodicity
            max_corner = max_corner .- min_corner
            min_corner = min_corner .- min_corner
        end

        # Note that we don't shift everything so that the first cell starts at `min_corner`.
        # The first cell is the cell containing `min_corner`, so we need to add one layer
        # in order for `max_corner` to be inside a cell.
        n_cells_per_dimension = ceil.(Int, (max_corner .- min_corner) ./ search_radius) .+ 1
        linear_indices = LinearIndices(Tuple(n_cells_per_dimension))
        min_cell = Tuple(floor_to_int.(min_corner ./ search_radius))

        cells = [Int32[] for _ in 1:prod(n_cells_per_dimension)]
    end

    return FullGridCellList{typeof(cells), typeof(linear_indices),
                            typeof(min_cell)}(cells, linear_indices, min_cell)
end

function Base.empty!(cell_list::FullGridCellList)
    Base.empty!.(cell_list.cells)

    return cell_list
end

function Base.empty!(cell_list::FullGridCellList{Nothing})
    # This is an empty "template" cell list to be used with `copy_cell_list`
    throw(UndefRefError("`search_radius` is not defined for this cell list"))
end

function push_cell!(cell_list::FullGridCellList, cell, particle)
    push!(cell_list[cell], particle)
end

function push_cell!(cell_list::FullGridCellList{Nothing}, cell, particle)
    # This is an empty "template" cell list to be used with `copy_cell_list`
    throw(UndefRefError("`search_radius` is not defined for this cell list"))
end

function deleteat_cell!(cell_list::FullGridCellList, cell, i)
    deleteat!(cell_list[cell], i)
end

@inline each_cell_index(cell_list::FullGridCellList) = eachindex(cell_list.cells)

function each_cell_index(cell_list::FullGridCellList{Nothing})
    # This is an empty "template" cell list to be used with `copy_cell_list`
    throw(UndefRefError("`search_radius` is not defined for this cell list"))
end

@inline function Base.getindex(cell_list::FullGridCellList, cell::Tuple)
    (; cells, linear_indices, min_cell) = cell_list

    return cells[linear_indices[(cell .- min_cell .+ 1)...]]
end

@inline function Base.getindex(cell_list::FullGridCellList, i::Integer)
    return cell_list.cells[i]
end

@inline function is_correct_cell(cell_list::FullGridCellList, cell_coords, cell_index)
    (; cells, linear_indices, min_cell) = cell_list

    return cells[linear_indices[(cell_coords .- min_cell .+ 1)...]] == cell_index
end

@inline index_type(::FullGridCellList) = Int32

function copy_cell_list(cell_list::FullGridCellList, search_radius, periodic_box)
    # Misuse `min_cell` to store min and max corner for copying
    min_corner, max_corner = cell_list.min_cell

    return FullGridCellList(; min_corner, max_corner, search_radius,
                            periodicity = !isnothing(periodic_box))
end
