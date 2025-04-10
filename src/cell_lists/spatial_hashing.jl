"""
    SpatialHashingCellList{NDIMS}(; list_size)

A basic spatial hashing implementation. Similar to [`DictionaryCellList`](@ref), the domain is discretized into cells, 
and the particles in each cell are stored in a hash map. The hash is computed using the spatial location of each cell 
[as described by Ihmsen et al. (2001)](@cite Ihmsen2003). By using a hash map, which only stores non-empty cells, 
the domain is effectively infinite. The size of the hash map is recommended to be approximately twice the number of particles balance memory consumption  
and the likelihood of hash collisions.

# Arguments
- `NDIMS::Int`: Number of spatial dimensions (e.g., `2` or `3`).
- `list_size::Int`: Size of the hash map (e.g., `2 * n_points`) .
"""
struct SpatialHashingCellList{CL, CI, CF}
    list_size::Int
    NDIMS::Int
    points::CL
    coords::CI
    collisions::CF
end

@inline index_type(::SpatialHashingCellList) = Int32

function supported_update_strategies(::PointNeighbors.SpatialHashingCellList)
    return (SerialUpdate, SemiParallelUpdate)
end

function SpatialHashingCellList{NDIMS}(list_size) where {NDIMS}
    points = [Int[] for _ in 1:list_size]
    collisions = [false for _ in 1:list_size]
    coords = [ntuple(_ -> typemin(Int), NDIMS) for _ in 1:list_size]
    return SpatialHashingCellList(list_size, NDIMS, points, coords,
                                  collisions)
end

function Base.empty!(cell_list::SpatialHashingCellList)
    (; list_size, NDIMS) = cell_list

    Base.empty!.(cell_list.points)
    cell_list.coords .= [ntuple(_ -> typemin(Int), NDIMS) for _ in 1:list_size]
    cell_list.collisions .= false
    return cell_list
end

function push_cell!(cell_list::SpatialHashingCellList, cell, point)
    (; points, coords, collisions, list_size, NDIMS) = cell_list
    key = spatial_hash(cell, list_size)
    cell_coord = coords[key]
    push!(points[key], point)
    # Check if a cell has been added at this hash
    if cell_coord == ntuple(_ -> typemin(Int), NDIMS)
        coords[key] = cell
    # Detect collision
    elseif cell_coord != cell
        collisions[key] = true
    end
end

# function push_cell!(cell_list::SpatialHashingCellList, cell)
#     (; coords, collisions, list_size, NDIMS) = cell_list
#     key = spatial_hash(cell, list_size)
#     cell_coord = coords[key]
#     if cell_coord == ntuple(_ -> typemin(Int), NDIMS)
#         coords[key] = cell
#     elseif cell_coord != cell
#         collisions[key] = true
#     end
# end

function deleteat_cell!(cell_list::SpatialHashingCellList, cell, i)
    deleteat!(cell_list[cell], i)
end

@inline each_cell_index(cell_list::SpatialHashingCellList) = eachindex(cell_list.points)

@inline function Base.getindex(cell_list::SpatialHashingCellList, cell::Tuple)
    (; points) = cell_list
    return points[spatial_hash(cell, length(points))]
end

@inline function Base.getindex(cell_list::SpatialHashingCellList, i::Integer)
    return cell_list.points[i]
end

@inline function is_correct_cell(cell_list::SpatialHashingCellList{<:Any, Nothing},
                                 coords, cell_index::Array)
    return coords == cell_index
end

function spatial_hash(cell::CartesianIndex, list_size)
    return spatial_hash(Tuple(cell), list_size)
end

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

function spatial_hash(cell::NTuple{3, Real}, index, list_size)
    i, j, k = cell

    return mod(xor(i * 73856093, j * 19349663, k * 83492791, index * 7238423947),
               list_size) + 1
end

