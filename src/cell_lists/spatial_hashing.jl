"""
    SpatialHashingCellList{NDIMS}(; list_size, 
                                    backend = DynamicVectorOfVectors{Int32},
                                    max_points_per_cell = 100)

A basic spatial hashing implementation. Similar to [`DictionaryCellList`](@ref), the domain is discretized into cells, 
and the particles in each cell are stored in a hash map. The hash is computed using the spatial location of each cell, 
as described by Ihmsen et al. (2011)(@cite Ihmsen2011). By using a hash map that stores entries only for non-empty cells, 
the domain is effectively infinite. The size of the hash map is recommended to be approximately twice the number of particles 
to balance memory consumption against the likelihood of hash collisions.

# Arguments
- `NDIMS::Int`: Number of spatial dimensions (e.g., `2` or `3`).
- `list_size::Int`: Size of the hash map (e.g., `2 * n_points`).
- `backend = DynamicVectorOfVectors{Int32}`: Type of the data structure to store the actual
    cell lists. Can be
    - `Vector{Vector{Int32}}`: Scattered memory, but very memory-efficient.
    - `DynamicVectorOfVectors{Int32}`: Contiguous memory, optimizing cache-hits.
- `max_points_per_cell = 100`: Maximum number of points per cell. This will be used to
                               allocate the `DynamicVectorOfVectors`. It is not used with
                               the `Vector{Vector{Int32}}` backend.
"""

struct SpatialHashingCellList{NDIMS, CL, CI, CF} <: AbstractCellList
    cells      :: CL
    coords     :: CI
    collisions :: CF
    list_size  :: Int

    # This constructor is necessary for Adapt.jl to work with this struct
    function SpatialHashingCellList(NDIMS, cells, coords, collisions, list_size)
        return new{NDIMS, typeof(cells),
                   typeof(coords), typeof(collisions)}(cells, coords,
                                                       collisions, list_size)
    end
end

@inline index_type(::SpatialHashingCellList) = Int32

@inline Base.ndims(::SpatialHashingCellList{NDIMS}) where {NDIMS} = NDIMS

function supported_update_strategies(::SpatialHashingCellList{T1, <:DynamicVectorOfVectors}) where {T1}
    return (ParallelUpdate, SerialUpdate)
end

function supported_update_strategies(::SpatialHashingCellList)
    return (SerialUpdate;)
end

function SpatialHashingCellList{NDIMS}(list_size,
                                       backend = DynamicVectorOfVectors{Int32},
                                       max_points_per_cell = 100) where {NDIMS}
    cells = construct_backend(backend, list_size,
                              max_points_per_cell)
    collisions = [false for _ in 1:list_size]
    coords = [typemin(Int) for _ in 1:list_size]

    return SpatialHashingCellList(NDIMS, cells, coords, collisions, list_size)
end

function Base.empty!(cell_list::SpatialHashingCellList)
    (; cells) = cell_list
    NDIMS = ndims(cell_list)

    # `Base.empty!.(cells)`, but for all backends
    @threaded default_backend(cells) for i in eachindex(cells)
        emptyat!(cells, i)
    end

    fill!(cell_list.coords, typemin(Int))
    cell_list.collisions .= false
    return cell_list
end

# For each entry in the hash table, store the coordinates of the cell of the first point being inserted at this entry.
# If a point with a different cell coordinate is being added, we have found a collision.
function push_cell!(cell_list::SpatialHashingCellList, cell, point)
    (; cells, coords, collisions, list_size) = cell_list
    NDIMS = ndims(cell_list)
    hash_key = spatial_hash(cell, list_size)

    @boundscheck check_cell_bounds(cell_list, hash_key)
    @inbounds pushat!(cells, hash_key, point)
    cell_coord_hash = coordinates_hash(cell)
    cell_coord = coords[hash_key]
    if cell_coord == typemin(Int)
        # If this cell is not used yet, set cell coordinates
        coords[hash_key] = cell_coord_hash
    elseif cell_coord != cell_coord_hash
        # If it is already used by a different cell, mark as collision
        collisions[hash_key] = true
    end
end

function push_cell_atomic!(cell_list::SpatialHashingCellList, cell, point)
    (; cells, coords, collisions, list_size) = cell_list
    NDIMS = ndims(cell_list)
    hash_key = spatial_hash(cell, list_size)

    @info cell
    cell_coord_hash = coordinates_hash(cell)

    @boundscheck check_cell_bounds(cell_list, hash_key)
    @inbounds pushat_atomic!(cells, hash_key, point)

    cell_coord = @inbounds coords[hash_key]
    if cell_coord == ntuple(_ -> typemin(Int), Val(NDIMS))
        # Throws `bitcast: value not a primitive type`-error
        # @inbounds Atomix.@atomic coords[hash_key] = cell
        # If this cell is not used yet, set cell coordinates
        @inbounds coords[hash_key] = cell_coord_hash
    elseif cell_coord != cell_coord_hash
        # If it is already used by a different cell, mark as collision
        @inbounds Atomix.@atomic collisions[hash_key] = true
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

@inline function check_cell_bounds(cell_list::SpatialHashingCellList, cell::Tuple)
    check_cell_bounds(cell_list, spatial_hash(cell, cell_list.list_size))
end

function coordinates_hash(cell_coordinate)
    # Check the dimensionality of the coordinate since we can not stuff more the 3 UInt32 in a UInt128
    @assert length(cell_coordinate <= 4)

    function coords2uint(hash::UInt128, coord::Int, n::Int)
        ua = reinterpret(UInt32, Int32(coord))
        return (UInt128(ua) << (n * 32)) | hash
    end

    hash = UInt128(0)
    for (i, coord) in enumerate(cell_coordinate)
        hash = coords2uint(hash, coord, i-1)
    end
    return hash
end

# function coordinates_hash_10(cell_coordinate)
#     shift10 = 0
#     hash = Int128(0)

#     function shift_by_10(x, n::Int)
#         @assert n >= 0
#         x = Int128(x)
#         for _ in 1:n
#             # multiply by 10 with binary shift operations
#             x = (x << 3) + (x << 1)
#         end
#         return x
#     end

#     for i in reverse(1:length(cell_coordinate))
#         coord = cell_coordinate[i]

#         # shift coord `shift` many times by 10 and add up
#         hash = hash + shift_by_10(coord, shift10)

#         # compute the shift for the next iteration
#         shift10 += length(string(abs(coord)))

#     end

#     return Int(hash)
# end
