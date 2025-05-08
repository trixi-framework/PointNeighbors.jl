"""
    SpatialHashingCellList{NDIMS}(; list_size)

A basic spatial hashing implementation. Similar to [`DictionaryCellList`](@ref), the domain is discretized into cells, 
and the particles in each cell are stored in a hash map. The hash is computed using the spatial location of each cell, 
as described by Ihmsen et al. (2011)(@cite Ihmsen2011). By using a hash map that stores entries only for non-empty cells, 
the domain is effectively infinite. The size of the hash map is recommended to be approximately twice the number of particles 
to balance memory consumption against the likelihood of hash collisions.

# Arguments
- `NDIMS::Int`: Number of spatial dimensions (e.g., `2` or `3`).
- `list_size::Int`: Size of the hash map (e.g., `2 * n_points`) .
"""

struct SpatialHashingCellList{NDIMS, CL, CI, CF} <: AbstractCellList
    cells      :: CL
    coords     :: CI
    collisions :: CF
    list_size  :: Int
end

@inline index_type(::SpatialHashingCellList) = Int32

@inline Base.ndims(::SpatialHashingCellList{NDIMS}) where {NDIMS} = NDIMS

function supported_update_strategies(::SpatialHashingCellList)
    return (SerialUpdate,)
end

function SpatialHashingCellList{NDIMS}(list_size,
                                       backend = DynamicVectorOfVectors{Int32},
                                       max_points_per_cell = 100) where {NDIMS}
    cells = construct_backend(SpatialHashingCellList, backend, list_size,
                              max_points_per_cell)
    collisions = [false for _ in 1:list_size]
    coords = [ntuple(_ -> typemin(Int), NDIMS) for _ in 1:list_size]

    return SpatialHashingCellList{NDIMS, typeof(cells), typeof(coords),
                                  typeof(collisions)}(cells, coords, collisions, list_size)
end

function construct_backend(::Type{SpatialHashingCellList}, ::Type{Vector{Vector{T}}}, size,
                           max_points_per_cell) where {T}
    return [T[] for _ in 1:size]
end

function construct_backend(::Type{SpatialHashingCellList},
                           ::Type{DynamicVectorOfVectors{T}}, size,
                           max_points_per_cell) where {T}
    cells = DynamicVectorOfVectors{T}(max_outer_length = size,
                                      max_inner_length = max_points_per_cell)
    # Do I still need that resize?  
    resize!(cells, size)

    return cells
end

function Base.empty!(cell_list::SpatialHashingCellList)
    (; list_size) = cell_list
    NDIMS = ndims(cell_list)

    # `Base.empty!.(cells)`, but for all backends
    for i in eachindex(cell_list.cells)
        emptyat!(cell_list.cells, i)
    end

    cell_list.coords .= [ntuple(_ -> typemin(Int), NDIMS) for _ in 1:list_size]
    cell_list.collisions .= false
    return cell_list
end

# For each entry in the hash table, store the coordinates of the cell of the first point being inserted at this entry.
# If a point with a different cell coordinate is being added, we have found a collision.
function push_cell!(cell_list::SpatialHashingCellList, cell, point)
    (; cells, coords, collisions, list_size) = cell_list
    NDIMS = ndims(cell_list)
    hash_key = spatial_hash(cell, list_size)

    # Do I need that @boundscheck?
    # @boundscheck check_cell_bounds(cell_list, cell)
    @inbounds pushat!(cells, hash_key, point)

    cell_coord = coords[hash_key]
    if cell_coord == ntuple(_ -> typemin(Int), NDIMS)
        # If this cell is not used yet, set cell coordinates
        coords[hash_key] = cell
    elseif cell_coord != cell
        # If it is already used by a different cell, mark as collision
        collisions[hash_key] = true
    end
end

function deleteat_cell!(cell_list::SpatialHashingCellList, cell, i)
    deleteat!(cell_list[cell], i)
end

@inline each_cell_index(cell_list::SpatialHashingCellList) = eachindex(cell_list.cells)

function copy_cell_list(cell_list::SpatialHashingCellList, search_radius,
                        periodic_box)
    (; list_size) = cell_list
    NDIMS = ndims(cell_list)

    # Here I'm using max_points_per_cell which is defined in src/cell_lists/full_grid.jl 
    # Think about putting it somewhere all cell list can access it or copying it here
    return SpatialHashingCellList{NDIMS}(list_size, typeof(cell_list.cells),
                                         max_points_per_cell(cell_list.cells))
end

@inline function Base.getindex(cell_list::SpatialHashingCellList, cell::Tuple)
    return cell_list.cells[spatial_hash(cell, length(cell_list.cells))]
end

@inline function Base.getindex(cell_list::SpatialHashingCellList, i::Integer)
    return cell_list.cells[i]
end

@inline function is_correct_cell(cell_list::SpatialHashingCellList{<:Any, Nothing},
                                 coords, cell_index::Array)
    return coords == cell_index
end

# Hash functions according to Ihmsen et al. (2001)
function spatial_hash(cell::NTuple{1, Real}, list_size)
    return mod(cell[1] * 73856093, list_size) + 1
end

function spatial_hash(cell::NTuple{2, Real}, list_size)
    i, j = cell

    return mod(xor(i * 73856093, j * 19349663), list_size) + 1
end

function spatial_hash(cell::NTuple{3, Real}, list_size)
    i, j, k = cell

    return mod(xor(i * 73856093, j * 19349663, k * 83492791), list_size) + 1
end
