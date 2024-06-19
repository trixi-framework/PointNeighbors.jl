struct FullGridCellList{C, LI, MC}
    cells          :: C
    linear_indices :: LI
    min_cell       :: MC

    function FullGridCellList{C, LI, MC}(cells, linear_indices, min_cell) where {C, LI, MC}
        new{C, LI, MC}(cells, linear_indices, min_cell)
    end
end

function FullGridCellList(min_corner, max_corner, search_radius; periodicity = false)
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

    cells = [Int[] for _ in 1:prod(n_cells_per_dimension)]

    return FullGridCellList{typeof(cells), typeof(linear_indices),
                            typeof(min_cell)}(cells, linear_indices, min_cell)
end

function Base.empty!(cell_list::FullGridCellList)
    Base.empty!.(cell_list.cells)

    return cell_list
end

function push_cell!(cell_list::FullGridCellList, cell, particle)
    push!(cell_list[cell], particle)
end

function deleteat_cell!(cell_list::FullGridCellList, cell, i)
    deleteat!(cell_list[cell], i)
end

@inline each_cell_index(cell_list::FullGridCellList) = eachindex(cell_list.cells)

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

@inline index_type(::FullGridCellList) = Int