"""
Tree-based data structure for storing cell lists. 
Based on `min_corner` and `max_corner`, we construct a covering grid. 
The cell size of this grid is `search_radius`. Assuming we are using
a particle refinement, where each particle has its own search radius,
the passed search radius should be the smallest search radius of all particles.
`cells_index` is an index array, `cells` stores the actual data. cells_index[i] contains an index to cells,
i.e. the data for cell i is stored at cells[cells_index[i]]. 
We use Morton indexing for efficient memory access and cells can be of different size. 
We populate the cell list, by iterating over all points, calculating the cartesian coordinates of the cell its in, 
convert to the Morton index with cartesian2morton() and insert it there.   
"""
struct TreeCellList{C, CIO, LI, MINC, MAXC} <: AbstractCellList
    cells          :: C
    cells_index :: CIO
    linear_indices :: LI
    min_corner     :: MINC
    max_corner     :: MAXC
end

@inline Base.ndims(cell_list::TreeCellList) = ndims(cell_list.linear_indices)

function supported_update_strategies(::TreeCellList{<:DynamicVectorOfVectors})
    return (SerialUpdate,)
end

function supported_update_strategies(::TreeCellList)
    return (SerialUpdate,)
end

function TreeCellList(; min_corner, max_corner, max_n_cells = nothing, 
                          search_radius = ntuplezero(eltype(min_corner)),
                          backend = DynamicVectorOfVectors{Int32},
                          max_points_per_cell = 100)
    if length(min_corner) != length(max_corner)
        throw(ArgumentError("min_corner and max_corner must have the same length"))
    end

    if length(min_corner) > 100
        throw(ArgumentError("TreeCellList only supports up to 100 dimensions, " *
                            "check your `min_corner` and `max_corner`"))
    end

    cells_index = Dict{UInt64, UnitRange{Int}}()   

    min_corner = SVector(Tuple(min_corner .- 1001 // 1000 * search_radius))
    max_corner = SVector(Tuple(max_corner .+ 1001 // 1000 * search_radius))
    
    if search_radius < eps()
        # Create an empty "template" cell list to be used with `copy_cell_list`
        cells = construct_backend(backend, 0, max_points_per_cell)
        linear_indices = LinearIndices(ntuple(_ -> 0, length(min_corner)))
        max_n_cells = 0     

    else
        n_cells_per_dimension = ceil.(Int, (max_corner .- min_corner) ./ search_radius)
        linear_indices = LinearIndices(Tuple(n_cells_per_dimension))


        if isnothing(max_n_cells)
            max_n_cells = 2 * prod(n_cells_per_dimension)
        end

        cells = construct_backend(backend, max_n_cells,
                                  max_points_per_cell)     
    end

    return TreeCellList(cells, cells_index, linear_indices, min_corner, max_corner)
end

@inline function cell_coords(coords, periodic_box::Nothing, cell_list::TreeCellList,
                             cell_size)
    (; min_corner) = cell_list

    # Subtract `min_corner` to offset coordinates so that the min corner of the grid
    # corresponds to the (1, 1, 1) cell.
    return cartesian2morton(Tuple(floor_to_int.((coords .- min_corner) ./ cell_size)) .+ 1)
end

function Base.empty!(cell_list::TreeCellList)
    (; cells) = cell_list

    # `Base.empty!.(cells)`, but for all backends
    @threaded default_backend(cells) for i in eachindex(cells)
        emptyat!(cells, i)
    end

    return cell_list
end

function Base.empty!(cell_list::TreeCellList{Nothing})
    # This is an empty "template" cell list to be used with `copy_cell_list`
    error("`search_radius` is not defined for this cell list")
end

function push_cell!(cell_list::TreeCellList, cell, particle)
    (; cells) = cell_list

    @boundscheck check_cell_bounds(cell_list, cell)

    # `push!(cell_list[cell], particle)`, but for all backends
    @inbounds pushat!(cells, cell_index(cell_list, cell), particle)

    return cell_list
end

function push_cell!(cell_list::TreeCellList{Nothing}, cell, particle)
    # This is an empty "template" cell list to be used with `copy_cell_list`
    error("`search_radius` is not defined for this cell list")
end

@inline function push_cell_atomic!(cell_list::TreeCellList, cell, particle)
    (; cells) = cell_list

    @boundscheck check_cell_bounds(cell_list, cell)

    # `push!(cell_list[cell], particle)`, but for all backends.
    # The atomic version of `pushat!` uses atomics to avoid race conditions when `pushat!`
    # is used in a parallel loop.
    @inbounds pushat_atomic!(cells, cell_index(cell_list, cell), particle)

    return cell_list
end

function deleteat_cell!(cell_list::TreeCellList, cell, i)
    (; cells) = cell_list

    @boundscheck check_cell_bounds(cell_list, cell)

    # `deleteat!(cell_list[cell], i)`, but for all backends
    deleteatat!(cells, cell_index(cell_list, cell), i)
end

@inline each_cell_index(cell_list::TreeCellList) = eachindex(cell_list.cells)

function each_cell_index(cell_list::TreeCellList{Nothing})
    # This is an empty "template" cell list to be used with `copy_cell_list`
    error("`search_radius` is not defined for this cell list")
end

@propagate_inbounds cell_index(cell_list::TreeCellList, cell::Tuple) = cartesian2morton(cell)

@inline cell_index(::TreeCellList, cell::Integer) = cell

@propagate_inbounds function Base.getindex(cell_list::TreeCellList, cell)
    (; cells) = cell_list

    return cells[cell_index(cell_list, cell)]
end

@inline function is_correct_cell(cell_list::TreeCellList, cell, cell_index_)
    @boundscheck check_cell_bounds(cell_list, cell)

    return cell_index(cell_list, cell) == cell_index_
end

@inline index_type(::TreeCellList) = Int32

function copy_cell_list(cell_list::TreeCellList, search_radius, periodic_box)
    (; min_corner, max_corner) = cell_list

    return TreeCellList(; min_corner, max_corner, search_radius,
                            backend = typeof(cell_list.cells),
                            max_points_per_cell = max_inner_length(cell_list.cells, 100))
end

@inline function check_cell_bounds(cell_list::TreeCellList{<:DynamicVectorOfVectors{<:Any,
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
@inline function check_cell_bounds(cell_list::TreeCellList, cell::Tuple)
    (; linear_indices) = cell_list

    # Make sure that points are not added to the outer padding layer, which is needed
    # to ensure that neighboring cells in all directions of all non-empty cells exist.
    if !all(cell[i] in 2:(size(linear_indices, i) - 1) for i in eachindex(cell))
        error("particle coordinates are NaN or outside the domain bounds of the cell list")
    end
end
