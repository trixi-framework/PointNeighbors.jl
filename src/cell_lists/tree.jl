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
struct TreeCellList{NDIMS, LI, MINC, MAXC} <: AbstractCellList
    linear_indices :: LI
    min_corner     :: MINC
    max_corner     :: MAXC

    particle_z       :: Vector{UInt64} # Calculate the z-index for each particle 
    particle_indices :: Vector{Int} # Contains the particle indices, will be sorted based on z-code. 
    cell_z           :: Vector{UInt64} # Unique z-indices of the cells that contain particles
    cell_ranges      :: Vector{UnitRange{Int}} # The index in the sorted particle array where this cell's particles begin.
    cell_levels      :: Vector{UInt8}

    min_cell_length   :: Float64
    grid_length       :: Float64
    max_depth         :: Int
    capacity_per_cell :: Int
end

function TreeCellList{NDIMS}(; min_corner, max_corner, n_particles, depth = 16,
                             capacity_per_cell = 1) where {NDIMS}
    n_cells_per_dimension = 2^depth
    min_corner = SVector(Tuple(min_corner .- 1001 // 1000 // n_cells_per_dimension))
    max_corner = SVector(Tuple(max_corner .+ 1001 // 1000 // n_cells_per_dimension))

    grid_length = maximum(max_corner - min_corner)
    min_cell_length = grid_length / n_cells_per_dimension
    linear_indices = LinearIndices(ntuple(_ -> n_cells_per_dimension, NDIMS))
    n_cells = n_cells_per_dimension^NDIMS

    cell_z = Vector{UInt64}(undef, 0)
    cell_ranges = Vector{UnitRange{Int}}(undef, 0)
    cell_levels = Vector{UInt8}(undef, 0)
    particle_z = Vector{UInt64}(undef, 0)
    particle_indices = Vector{Int}(undef, 0)

    return TreeCellList{NDIMS, typeof(linear_indices), typeof(min_corner),
                        typeof(max_corner)}(linear_indices, min_corner, max_corner,
                                            particle_z, particle_indices, cell_z,
                                            cell_ranges, cell_levels, min_cell_length,
                                            grid_length, depth, capacity_per_cell)
end

function Base.empty!(cell_list::TreeCellList)
    (; particle_z, particle_indices, cell_z, cell_ranges, cell_levels) = cell_list
    empty!(particle_z)
    empty!(particle_indices)
    empty!(cell_z)
    empty!(cell_ranges)
    empty!(cell_levels)

    return cell_list
end

# TODO
function push_cell!(cell_list::TreeCellList, cell, point, point_coords)
    return cell_list
end

@inline each_cell_index(cell_list::TreeCellList) = eachindex(cell_list.cell_z)

function each_cell_index(cell_list::TreeCellList{Nothing})
    # This is an empty "template" cell list to be used with `copy_cell_list`
    error("`search_radius` is not defined for this cell list")
end

@propagate_inbounds function cell_index(cell_list::TreeCellList, cell::Tuple)
    (; linear_indices) = cell_list

    return linear_indices[cell...]
end

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
