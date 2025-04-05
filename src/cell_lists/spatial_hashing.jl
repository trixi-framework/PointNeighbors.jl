struct SpatialHashingCellList{CL, CI, CF}
    list_size::Int
    NDIMS::Int
    cell_points::CL
    cell_coords::CI # Think about renaming this field, there is a function with the same name
    cell_collision::CF
end

@inline index_type(::SpatialHashingCellList) = Int64

# Change the order back to (SemiParallelUpdate, SerialUpdate) when SemiParallelUpdate is implemented
function supported_update_strategies(::SpatialHashingCellList)
    return (SerialUpdate, SemiParallelUpdate)
end

function SpatialHashingCellList{NDIMS}(list_size) where {NDIMS}
    cell_points = [Int[] for _ in 1:list_size]
    cell_collision = [false for _ in 1:list_size]

    # Field cell_choords is used to check if there is a collision and contains
    # the coordinates of the first cell added to the hash list
    cell_coords = [ntuple(_ -> typemin(Int), NDIMS) for _ in 1:list_size]
    return SpatialHashingCellList(list_size, NDIMS, cell_points, cell_coords,
                                  cell_collision)
end

function Base.empty!(cell_list::SpatialHashingCellList)
    (; list_size, NDIMS) = cell_list

    Base.empty!.(cell_list.cell_points)
    cell_list.cell_coords .= [ntuple(_ -> typemin(Int), NDIMS) for _ in 1:list_size]
    cell_list.cell_collision .= false
    return cell_list
end

function push_cell!(cell_list::SpatialHashingCellList, cell, point)
    (; cell_points, cell_coords, cell_collision, list_size) = cell_list
    key = spatial_hash(cell, list_size)
    cell_coord = cell_coords[key]
    NDIMS = length(cell)

    push!(cell_points[key], point)
    # Check if the a cell has been added at this hash
    if cell_coord == ntuple(_ -> typemin(Int), NDIMS)
        cell_coords[key] = cell
    # Detect collision
    elseif cell_coord != cell
        cell_collision[key] = true
    end
end

# Implement reset of collision flag, if after the deletion there still is no collision?
# Not needed if we do not use update!() but initialize!()
function deleteat_cell!(cell_list::SpatialHashingCellList, cell, i)
    deleteat!(cell_list[cell], i)
end

@inline each_cell_index(cell_list::SpatialHashingCellList) = eachindex(cell_list.cell_points)

@inline function Base.getindex(cell_list::SpatialHashingCellList, cell::Tuple)
    (; cell_points) = cell_list

    return cell_points[spatial_hash(cell, length(cell_points))]
end

@inline function Base.getindex(cell_list::SpatialHashingCellList, i::Integer)
    return cell_list.cell_points[i]
end

# Naming of cell_coords and cell_index confusing
@inline function is_correct_cell(cell_list::SpatialHashingCellList{<:Any, Nothing},
                                 cell_coords, cell_index::Array)
    return cell_coords == cell_index
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

function spatial_hash(cell::CartesianIndex{2}, list_size)
    return spatial_hash(Tuple(cell), list_size)
end

function spatial_hash(cell::CartesianIndex{3}, list_size)
    return spatial_hash(Tuple(cell), list_size)
end
