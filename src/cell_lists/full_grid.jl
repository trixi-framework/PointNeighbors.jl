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
    return (ParallelIncrementalUpdate, ParallelUpdate, SemiParallelUpdate,
            SerialIncrementalUpdate, SerialUpdate)
end

function supported_update_strategies(::FullGridCellList)
    return (SemiParallelUpdate, SerialIncrementalUpdate, SerialUpdate)
end

function FullGridCellList(; min_corner, max_corner,
                          search_radius = zero(eltype(min_corner)),
                          backend = DynamicVectorOfVectors{Int32},
                          max_points_per_cell = 100)
    # Add one layer in each direction to make sure neighbor cells exist.
    # Also pad domain a little more to avoid 0 in cell indices due to rounding errors.
    # We can't just use `eps()`, as one might use lower precision types.
    # This padding is safe, and will give us one more layer of cells in the worst case.
    # `1001 // 1000` is 1.001 without forcing a float type.
    min_corner = SVector(Tuple(min_corner .- 1001 // 1000 * search_radius))
    max_corner = SVector(Tuple(max_corner .+ 1001 // 1000 * search_radius))

    if search_radius < eps()
        # Create an empty "template" cell list to be used with `copy_cell_list`
        cells = construct_backend(FullGridCellList, backend, 0, max_points_per_cell)
        linear_indices = LinearIndices(ntuple(_ -> 0, length(min_corner)))
    else
        n_cells_per_dimension = ceil.(Int, (max_corner .- min_corner) ./ search_radius)
        linear_indices = LinearIndices(Tuple(n_cells_per_dimension))

        cells = construct_backend(FullGridCellList, backend, prod(n_cells_per_dimension),
                                  max_points_per_cell)
    end

    return FullGridCellList(cells, linear_indices, min_corner, max_corner)
end

@inline function cell_coords(coords, periodic_box::Nothing, cell_list::FullGridCellList,
                             cell_size)
    (; min_corner) = cell_list

    # Subtract `min_corner` to offset coordinates so that the min corner of the grid
    # corresponds to the (1, 1, 1) cell.
    return Tuple(floor_to_int.((coords .- min_corner) ./ cell_size)) .+ 1
end

@inline function cell_coords(coords, periodic_box::PeriodicBox, cell_list::FullGridCellList,
                             cell_size)
    # Subtract `periodic_box.min_corner` to offset coordinates so that the min corner
    # of the grid corresponds to the (0, 0, 0) cell.
    offset_coords = periodic_coords(coords, periodic_box) .- periodic_box.min_corner

    # Add 2, so that the min corner will be the (2, 2, 2)-cell.
    # With this, we still have one padding layer in each direction around the periodic box,
    # just like without using a periodic box.
    # This is not needed for finding neighbor cells, but to make the bounds check
    # work the same way as without a periodic box.
    return Tuple(floor_to_int.(offset_coords ./ cell_size)) .+ 2
end

@inline function periodic_cell_index(cell_index, ::PeriodicBox, n_cells,
                                     cell_list::FullGridCellList)
    # 2-based modulo to match the indexing of the periodic box explained above.
    return mod.(cell_index .- 2, n_cells) .+ 2
end

function Base.empty!(cell_list::FullGridCellList)
    (; cells) = cell_list

    # `Base.empty!.(cells)`, but for all backends
    @threaded default_backend(cells) for i in eachindex(cells)
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

    @boundscheck check_cell_bounds(cell_list, cell)

    # `push!(cell_list[cell], particle)`, but for all backends
    @inbounds pushat!(cells, cell_index(cell_list, cell), particle)

    return cell_list
end

function push_cell!(cell_list::FullGridCellList{Nothing}, cell, particle)
    # This is an empty "template" cell list to be used with `copy_cell_list`
    error("`search_radius` is not defined for this cell list")
end

@inline function push_cell_atomic!(cell_list::FullGridCellList, cell, particle)
    (; cells) = cell_list

    @boundscheck check_cell_bounds(cell_list, cell)

    # `push!(cell_list[cell], particle)`, but for all backends.
    # The atomic version of `pushat!` uses atomics to avoid race conditions when `pushat!`
    # is used in a parallel loop.
    @inbounds pushat_atomic!(cells, cell_index(cell_list, cell), particle)

    return cell_list
end

function deleteat_cell!(cell_list::FullGridCellList, cell, i)
    (; cells) = cell_list

    @boundscheck check_cell_bounds(cell_list, cell)

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

@inline function is_correct_cell(cell_list::FullGridCellList, cell, cell_index_)
    @boundscheck check_cell_bounds(cell_list, cell)

    return cell_index(cell_list, cell) == cell_index_
end

@inline index_type(::FullGridCellList) = Int32

function copy_cell_list(cell_list::FullGridCellList, search_radius, periodic_box)
    (; min_corner, max_corner) = cell_list

    return FullGridCellList(; min_corner, max_corner, search_radius,
                            backend = typeof(cell_list.cells),
                            max_points_per_cell = max_points_per_cell(cell_list.cells))
end

@inline function check_cell_bounds(cell_list::FullGridCellList{<:DynamicVectorOfVectors{<:Any,
                                                                                        <:Array}},
                                   cell::Tuple)
    (; linear_indices) = cell_list

    # Make sure that points are not added to the outer padding layer, which is needed
    # to ensure that neighboring cells in all directions of all non-empty cells exist.
    if !all(cell[i] in 2:(size(linear_indices, i) - 1) for i in eachindex(cell))
        size_ = [2:(size(linear_indices, i) - 1) for i in eachindex(cell)]
        print_size_ = "[$(join(size_, ", "))]"
        error("particle coordinates are NaN or outside the domain bounds of the cell list\n" *
              "cell $cell is out of bounds for cell grid of size $print_size_")
    end
end

# On GPUs, we can't throw a proper error message because string interpolation is not
# allowed. Note that we cannot dispatch on `AbstractGPUArray`, as we are inside a kernel,
# so the array types are something like `CuDeviceArray`, which is not an `AbstractGPUArray`.
@inline function check_cell_bounds(cell_list::FullGridCellList, cell::Tuple)
    (; linear_indices) = cell_list

    # Make sure that points are not added to the outer padding layer, which is needed
    # to ensure that neighboring cells in all directions of all non-empty cells exist.
    if !all(cell[i] in 2:(size(linear_indices, i) - 1) for i in eachindex(cell))
        error("particle coordinates are NaN or outside the domain bounds of the cell list")
    end
end
