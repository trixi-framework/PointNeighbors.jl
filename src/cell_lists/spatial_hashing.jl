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
    points     :: CL
    coords     :: CI
    collisions :: CF
    list_size  :: Int
end

@inline index_type(::SpatialHashingCellList) = Int32

@inline Base.ndims(::SpatialHashingCellList{NDIMS}) where {NDIMS} = NDIMS

function supported_update_strategies(::SpatialHashingCellList)
    return (SerialUpdate,)
end

function SpatialHashingCellList{NDIMS}(list_size) where {NDIMS}
    points = [Int[] for _ in 1:list_size]
    collisions = [false for _ in 1:list_size]
    coords = [ntuple(_ -> typemin(Int), NDIMS) for _ in 1:list_size]
    return SpatialHashingCellList{NDIMS, typeof(points), typeof(coords),
                                  typeof(collisions)}(points, coords, collisions, list_size)
end

function Base.empty!(cell_list::SpatialHashingCellList)
    (; list_size) = cell_list
    NDIMS = ndims(cell_list)

    Base.empty!.(cell_list.points)
    cell_list.coords .= [ntuple(_ -> typemin(Int), NDIMS) for _ in 1:list_size]
    cell_list.collisions .= false
    return cell_list
end

# For each entry in the hash table, store the coordinates of the cell of the first point being inserted at this entry.
# If a point with a different cell coordinate is being added, we have found a collision.
function push_cell!(cell_list::SpatialHashingCellList, cell, point)
    (; points, coords, collisions, list_size) = cell_list
    NDIMS = ndims(cell_list)
    hash_key = spatial_hash(cell, list_size)
    push!(points[hash_key], point)

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

@inline each_cell_index(cell_list::SpatialHashingCellList) = eachindex(cell_list.points)

function copy_cell_list(cell_list::SpatialHashingCellList, search_radius,
                        periodic_box, n_points)
    (; list_size) = cell_list
    NDIMS = ndims(cell_list)

    return SpatialHashingCellList{NDIMS}(list_size)
end

@inline function Base.getindex(cell_list::SpatialHashingCellList, cell::Tuple)
    return cell_list.points[spatial_hash(cell, length(cell_list.points))]
end

@inline function Base.getindex(cell_list::SpatialHashingCellList, i::Integer)
    return cell_list.points[i]
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
