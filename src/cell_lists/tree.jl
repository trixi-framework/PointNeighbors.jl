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
struct TreeCellList{NDIMS, LI, MINC, MAXC, AC, CS} <: AbstractCellList
    linear_indices :: LI
    min_corner     :: MINC
    max_corner     :: MAXC

    active_cells :: AC
    marked_cells :: BitVector
    cell_levels  :: Vector{Int8}
    cell_sizes   :: CS

    min_cell_length   :: Float64
    grid_length       :: Float64
    max_level         :: Int
    capacity_per_cell :: Int
end

function TreeCellList{NDIMS}(; min_corner, max_corner, max_level = 16, backend = DynamicVectorOfVectors{Int32},
                               max_points_per_cell = 100, buffer_size = 10000, capacity_per_cell=100) where {NDIMS}
    n_cells_per_dimension = 2^max_level
    min_corner = SVector(Tuple(min_corner .- 1001 // 1000 // n_cells_per_dimension))
    max_corner = SVector(Tuple(max_corner .+ 1001 // 1000 // n_cells_per_dimension))

    grid_length = maximum(max_corner - min_corner)
    min_cell_length = grid_length / n_cells_per_dimension
    linear_indices = LinearIndices(ntuple(_ -> n_cells_per_dimension, NDIMS))
    n_cells = n_cells_per_dimension^NDIMS

    active_cells = construct_backend(backend, n_cells, max_points_per_cell)
    marked_cells = falses(n_cells)
    cell_levels = Vector{Int8}(undef, n_cells)
    cell_sizes = SVector(Tuple([min_cell_length * 2 ^ (max_level - i) for i in 0:max_level]))

    return TreeCellList{NDIMS, typeof(linear_indices), typeof(min_corner),
                        typeof(max_corner), typeof(active_cells), typeof(cell_sizes)}(linear_indices,
                                                                                      min_corner,
                                                                                      max_corner,
                                                                                      active_cells,
                                                                                      marked_cells,
                                                                                      cell_levels,
                                                                                      cell_sizes,
                                                                                      min_cell_length,
                                                                                      grid_length,
                                                                                      max_level,
                                                                                      capacity_per_cell)
end

function supported_update_strategies(::TreeCellList)
    return (ParallelUpdate, SerialUpdate)
end

function Base.empty!(cell_list::TreeCellList)
    (; active_cells,  marked_cells, cell_levels) = cell_list
    marked_cells .= false 
    cell_levels .= -1
    
    @threaded default_backend(active_cells) for i in eachindex(active_cells)
        emptyat!(active_cells, i)
    end


    return cell_list
end

function push_cell!(cell_list::TreeCellList, cell, particle)
    (; active_cells) = cell_list

    # TODO
    # @boundscheck check_cell_bounds(cell_list, cell)

    @inbounds pushat!(active_cells, cell_index(cell_list, cell), particle)

    return cell_list
end

function deleteat_cell!(cell_list::TreeCellList, cell, i)
    (; active_cells) = cell_list

    # TODO
    # @boundscheck check_cell_bounds(cell_list, cell)

    # `deleteat!(cell_list[cell], i)`, but for all backends
    deleteatat!(active_cells, cell_index(cell_list, cell), i)
end

@inline each_cell_index(cell_list::TreeCellList) = eachindex(cell_list.active_cells)

function each_cell_index(cell_list::TreeCellList{Nothing})
    # This is an empty "template" cell list to be used with `copy_cell_list`
    error("`search_radius` is not defined for this cell list")
end

@propagate_inbounds function cell_index(::TreeCellList, cell::Tuple)
    return morton2cartesian(collect(cell))
end

@inline cell_index(::TreeCellList, cell::Integer) = cell

@propagate_inbounds function Base.getindex(cell_list::TreeCellList, cell)
    (; active_cells) = cell_list

    return active_cells[cell_index(cell_list, cell)]
end

@inline function is_correct_cell(cell_list::TreeCellList, cell, cell_index_)
    # TODO
    # @boundscheck check_cell_bounds(cell_list, cell)

    return cell_index(cell_list, cell) == cell_index_
end

function is_leaf(cell_list, cell)
    return cell_list.cell_levels[cell] > -1
end

@inline index_type(::TreeCellList) = Int32

function copy_cell_list(cell_list::TreeCellList, search_radius, periodic_box)
    (; min_corner, max_corner) = cell_list

    return TreeCellList(; min_corner, max_corner, search_radius,
                        backend = typeof(cell_list.cells),
                        max_points_per_cell = max_inner_length(cell_list.active_cells, 100))
end

@inline function check_cell_bounds(cell_list::TreeCellList{<:DynamicVectorOfVectors{<:Any,<:Array}},
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
