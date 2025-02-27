struct SpatialHashingCellList{CL, CI, CF}
    cell_points       :: CL
    cell_coords :: CI
    cell_collision :: CF
    size        :: Int
end

@inline index_type(::SpatialHashingCellList) = Int64

# Change the order back to (SemiParallelUpdate, SerialUpdate) when SemiParallelUpdate is implemented
function supported_update_strategies(::SpatialHashingCellList)
    return (SerialUpdate, SemiParallelUpdate)
end

function SpatialHashingCellList{NDIMS}(size; handle_collisions = true) where {NDIMS}
    cell_points = [Int[] for _ in 1:size]
    cell_collision = [false for _ in 1:size]
    cell_coords = [NTuple{NDIMS, Int}[] for _ in 1:size]
    return SpatialHashingCellList(cell_points, cell_coords, cell_collision, size)
end

function Base.empty!(cell_list::SpatialHashingCellList)
    Base.empty!.(cell_list.cell_points)
    Base.empty!.(cell_list.cell_coords)
    cell_list.cell_collision .= false

    return cell_list
end

function push_cell!(cell_list::SpatialHashingCellList, cell, point)
    (; cell_points, cell_coords, cell_collision, size) = cell_list
    key = spatial_hash(cell, size)
    push!(cell_points[key], point)

    if !(cell in cell_coords[key])
        push!(cell_coords[key], cell)
    end

    if length(cell_coords[key]) > 1
        cell_collision[key] = true
    end
end

function get_points(nhs, cell, coords_fun)
    (; cell_list) = nhs
    (; cell_points, cell_coords, cell_collision, size) = cell_list
    key = spatial_hash(cell, size)
    if cell_collision[key]
        points_in_cell = []
        for point in cell_list.cell_points[key]
            if cell_coords(coords_fun(point), nhs) == cell
                push!(points_in_cell, point)
            end
        end
        return points_in_cell

    else
        return cell_list.cell_points[key]
    end

end

function deleteat_cell!(cell_list::SpatialHashingCellList, cell, i)
    deleteat!(cell_list[cell], i)
end

@inline each_cell_index(cell_list::SpatialHashingCellList) = eachindex(cell_list.cell_points)

@inline function Base.getindex(cell_list::SpatialHashingCellList, i)
    if isa(i, Int)
        return cell_list.cell_points[i]
    elseif isa(i, Tuple)
        return cell_list.cell_points[i...]
    end
end

@inline function is_correct_cell(cell_list::SpatialHashingCellList{<:Any, Nothing},
    cell_coords, cell_index::Array) # What is the correct type for cell_index? It is a list of integers
return cell_coords in cell_list.cell_coords[cell_index]
end

function spatial_hash(cell::NTuple{1, Real}, size)
    return mod(cell[1] * 73856093, size) + 1
end

function spatial_hash(cell::NTuple{2, Real}, size)
    i, j = cell

    return mod(xor(i * 73856093, j * 19349663), size) + 1
end

function spatial_hash(cell::NTuple{3, Real}, size)
    i, j, k = cell

    return mod(xor(i * 73856093, j * 19349663, k * 83492791), size) + 1
end

function spatial_hash(cell::NTuple{3, Real}, index, size)
    i, j, k = cell

    return mod(xor(i * 73856093, j * 19349663, k * 83492791, index * 7238423947), size) + 1
end
