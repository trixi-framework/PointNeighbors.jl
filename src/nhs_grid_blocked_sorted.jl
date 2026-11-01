# Specialization of the SIMD-vectorized neighbor loop (see `nhs_grid_blocked.jl`)
# for points sorted by cell (see `sorted_neighborhood_search`).
#
# With sorted points, the points of each block are contiguous in memory. The filter can then
# load the coordinates of all points of a block with a single SIMD load per dimension
# (with coordinates in a structure-of-arrays layout) instead of loading them through the
# cell list. Instead of bit masks, the filter writes a list of neighbor blocks
# for each point of a block, which is faster than the bit masks for sorted points.
# Like in `nhs_grid_blocked.jl`, this is only implemented for the point set pairs of a
# `CombinedGridNeighborhoodSearch`, and the points of the outer loop are taken from the
# cell list of the system points.

# A block of at most `BLOCK_SIZE` consecutive points
struct NeighborBlock
    first_point :: Int
    n_points    :: Int32
end

# The neighbors of a single point, which are stored as a list of `NeighborBlock`s
struct SortedNeighborBlocksOfPoint{NC, NHS, P, PC, B}
    neighbor_coords     :: NC
    neighborhood_search :: NHS
    point               :: P
    point_coords        :: PC
    neighbor_blocks     :: B   # Matrix with one column of neighbor blocks per lane
    lane                :: Int # Column of `neighbor_blocks` belonging to `point`
    n_blocks            :: Int # Number of neighbor blocks of `point`
end

# A `GridNeighborhoodSearch` for points sorted by cell (see `sorted_neighborhood_search`)
# on CPUs without periodicity
const SortedGridNHS = GridNeighborhoodSearch{<:Any, <:Any,
                                             <:FullGridCellList{<:CompactVectorOfVectors{<:Any,
                                                                                         <:Base.OneTo,
                                                                                         <:Array}},
                                             <:Any, Nothing}

# The points of the outer loop are taken from the cell list of the system point set.
@inline function foreach_point_inner(f, system_coords, neighbor_coords,
                                     neighborhood_search::PointSetPairSearch{<:SortedGridNHS},
                                     parallelization_backend::AbstractThreadingBackend,
                                     points, safe::Val, simd::Val{true})
    (; system_search, neighbor_search) = neighborhood_search

    foreach_point_blocked(f, system_coords, neighbor_coords, system_search,
                          neighbor_search, parallelization_backend, points, safe)
end

# Specialization of `foreach_point_blocked_inner` for sorted points.
# Returns the number of points for which `f` has been called.
function foreach_point_blocked_inner(f::T, system_coords, neighbor_coords,
                                     system_search::SortedGridNHS,
                                     neighbor_search::SortedGridNHS,
                                     parallelization_backend, point_mask,
                                     ::Val{SAFE}) where {T, SAFE}
    system_cells = system_search.cell_list.cells
    neighbor_cells = neighbor_search.cell_list.cells
    NDIMS = ndims(neighbor_search)

    # The points of the cell lists are simply `1:n` for sorted points.
    # With these checks, we can skip all bounds checks for loading coordinates.
    # The SIMD loads in `mapreduce_neighbor` load at least `BLOCK_SIZE` neighbor points.
    if SAFE
        checkbounds(system_coords, NDIMS, Base.OneTo(n_points_in_cells(system_cells)))
        checkbounds(neighbor_coords, NDIMS,
                    Base.OneTo(max(n_points_in_cells(neighbor_cells), BLOCK_SIZE)))
    end

    # One matrix of neighbor blocks per task with one column for each lane of a block
    max_neighbor_blocks = 3^NDIMS * cld(max_points_in_cells(neighbor_cells), BLOCK_SIZE)
    neighbor_blocks = [Matrix{NeighborBlock}(undef, max_neighbor_blocks, BLOCK_SIZE)
                       for _ in 1:n_tasks(parallelization_backend)]
    n_points_task = zeros(Int, n_tasks(parallelization_backend))

    # Each cell only writes to its own points, so all cells can be processed in parallel.
    foreach_cell_chunk(system_search, parallelization_backend) do task, cell_range
        n_points = foreach_point_in_cells_sorted(f, system_coords, neighbor_coords,
                                                 system_search, neighbor_search,
                                                 cell_range,
                                                 @inbounds(neighbor_blocks[task]),
                                                 point_mask, Val(SAFE))
        @inbounds n_points_task[task] += n_points
    end

    return sum(n_points_task)
end

# Points that are not assigned to a cell (inactive points) are stored at the end.
n_points_in_cells(cells::CompactVectorOfVectors) = last(cells.first_bin_index) - 1

# Returns the number of points for which `f` has been called.
@inline function foreach_point_in_cells_sorted(f, system_coords, neighbor_coords,
                                               system_search, neighbor_search, cell_range,
                                               neighbor_blocks, point_mask, safe)
    system_cells = system_search.cell_list.cells
    (; linear_indices) = neighbor_search.cell_list
    NDIMS = ndims(neighbor_search)
    offsets = neighbor_cell_offsets(linear_indices, Val(NDIMS))
    n_points_computed = 0

    for cell in cell_range
        n_points_cell = @inbounds n_points_in_cell(system_cells, cell)
        first_point_cell = @inbounds first_entry_in_cell(system_cells, cell)

        for first_point in
            first_point_cell:BLOCK_SIZE:(first_point_cell + n_points_cell - 1)

            n_points_block = min(BLOCK_SIZE, first_point_cell + n_points_cell - first_point)
            n_blocks = find_neighbor_blocks!(neighbor_blocks, system_coords,
                                             neighbor_coords, neighbor_search, cell,
                                             first_point, n_points_block, offsets, safe)

            for lane in 1:n_points_block
                point = first_point + lane - 1

                # Skip points of the cell list that are not in `points`.
                # `point` has been bounds checked in `foreach_point_blocked_inner`.
                is_in_points(point_mask, point) || continue
                n_points_computed += 1

                point_coords = @inbounds extract_svector(system_coords, Val(NDIMS), point)

                neighbors = SortedNeighborBlocksOfPoint(neighbor_coords, neighbor_search,
                                                        point, point_coords,
                                                        neighbor_blocks,
                                                        lane, Int(@inbounds n_blocks[lane]))
                @inline f(point, neighbors)
            end
        end
    end

    return n_points_computed
end

# Find the neighbor blocks that contain neighbors for all lanes of the block of
# `n_points_block` points starting at `first_point` at once.
# Return the number of neighbor blocks for each lane.
@inline function find_neighbor_blocks!(neighbor_blocks, system_coords, neighbor_coords,
                                       neighborhood_search, cell, first_point,
                                       n_points_block, offsets, safe)
    # The SIMD version loads the coordinates of all lanes of the block at once,
    # which is not possible for the final block of the points.
    if simd_loads_possible(system_coords) && simd_loads_possible(neighbor_coords) &&
       first_point + BLOCK_SIZE - 1 <= size(system_coords, 2)
        return find_neighbor_blocks_simd!(neighbor_blocks, system_coords, neighbor_coords,
                                          neighborhood_search, cell, first_point, offsets,
                                          safe)
    end

    return find_neighbor_blocks_generic!(neighbor_blocks, system_coords, neighbor_coords,
                                         neighborhood_search, cell, first_point,
                                         n_points_block, offsets, safe)
end

# The SIMD version requires coordinates that are stored contiguously for each dimension
# (structure-of-arrays layout), like a `PermutedDimsArray` of an `n_points × NDIMS` matrix.
@inline function simd_loads_possible(coords::Union{DenseMatrix{T},
                                                   PermutedDimsArray{T, 2}}) where {T <:
                                                                                    Union{Float32,
                                                                                          Float64}}
    return stride(coords, 2) == 1
end

@inline simd_loads_possible(coords) = false

# Same as `find_neighbor_blocks_generic!`, but with explicit SIMD instructions,
# which allows to keep everything in registers.
# The coordinates of all `BLOCK_SIZE` lanes are loaded with a single SIMD load
# per dimension. Lanes outside of the block contain the following points and are ignored.
@inline function find_neighbor_blocks_simd!(neighbor_blocks, system_coords,
                                            neighbor_coords, neighborhood_search, cell,
                                            first_point, offsets, ::Val{SAFE}) where {SAFE}
    (; cells) = neighborhood_search.cell_list
    NDIMS = ndims(neighborhood_search)
    T = eltype(neighbor_coords)
    ELTYPE = eltype(neighborhood_search)
    search_radius2 = filter_radius2(neighborhood_search)

    # Offset between the coordinates of two dimensions of the same point.
    system_dim_stride = stride(system_coords, 1)
    dim_stride = stride(neighbor_coords, 1)
    n_rows = size(neighbor_blocks, 1)
    n_blocks = zero(Vec{BLOCK_SIZE, Int32})

    GC.@preserve system_coords neighbor_coords begin
        system_ptr = pointer(system_coords)
        ptr = pointer(neighbor_coords)
        block_coords = ntuple(Val(NDIMS)) do dim
            return vload(Vec{BLOCK_SIZE, T},
                         system_ptr +
                         sizeof(T) * (first_point - 1 + (dim - 1) * system_dim_stride))
        end

        for offset in offsets
            neighbor_cell = cell + offset
            SAFE && checkbounds(cells, neighbor_cell)
            n_points_neighbor_cell = @inbounds n_points_in_cell(cells, neighbor_cell)
            first_point_neighbor_cell = @inbounds first_entry_in_cell(cells, neighbor_cell)
            last_point_neighbor_cell = first_point_neighbor_cell + n_points_neighbor_cell -
                                       1

            for neighbor_first_point in
                first_point_neighbor_cell:BLOCK_SIZE:last_point_neighbor_cell

                neighbor_last_point = min(neighbor_first_point + BLOCK_SIZE - 1,
                                          last_point_neighbor_cell)

                # Check for all lanes at once if any point of the neighbor block
                # is a neighbor
                has_neighbor = zero(Vec{BLOCK_SIZE, Int32})
                for neighbor in neighbor_first_point:neighbor_last_point
                    distance2 = zero(Vec{BLOCK_SIZE, ELTYPE})
                    for dim in 1:NDIMS
                        # All points have been bounds checked
                        # in `foreach_point_blocked_inner`.
                        neighbor_coord = unsafe_load(ptr, neighbor + (dim - 1) * dim_stride)

                        # Like in `mapreduce_neighbor`, convert to the type of the search
                        # radius after computing the difference
                        pos_diff = convert(Vec{BLOCK_SIZE, ELTYPE},
                                           block_coords[dim] - neighbor_coord)
                        distance2 = muladd(pos_diff, pos_diff, distance2)
                    end
                    has_neighbor |= vifelse(distance2 <= search_radius2,
                                            one(Vec{BLOCK_SIZE, Int32}),
                                            zero(Vec{BLOCK_SIZE, Int32}))
                end

                # Always write the block, but only keep it when it contains a neighbor.
                # Use a linear index with the number of rows from above, because
                # the compiler would otherwise reload it after each store.
                n_points_neighbor_block = neighbor_last_point - neighbor_first_point + 1
                block = NeighborBlock(neighbor_first_point, n_points_neighbor_block % Int32)
                for lane in 1:BLOCK_SIZE
                    @inbounds neighbor_blocks[(lane - 1) * n_rows + n_blocks[lane] + 1] = block
                end
                n_blocks += has_neighbor
            end
        end
    end

    return SVector(Tuple(n_blocks))
end

# Generic version of `find_neighbor_blocks!` for all coordinate arrays
@inline function find_neighbor_blocks_generic!(neighbor_blocks, system_coords,
                                               neighbor_coords, neighborhood_search, cell,
                                               first_point, n_points_block, offsets,
                                               ::Val{SAFE}) where {SAFE}
    (; cells) = neighborhood_search.cell_list
    NDIMS = ndims(neighborhood_search)
    search_radius2 = filter_radius2(neighborhood_search)

    # Load the coordinates of the block into a lane-contiguous buffer.
    # Unused lanes of the final block of a cell repeat the last point.
    block_coords = MMatrix{BLOCK_SIZE, NDIMS, eltype(system_coords)}(undef)
    for lane in 1:BLOCK_SIZE
        point = first_point + min(lane, n_points_block) - 1
        for dim in 1:NDIMS
            # All points have been bounds checked in `foreach_point_blocked_inner`.
            @inbounds block_coords[lane, dim] = system_coords[dim, point]
        end
    end

    n_blocks = MVector{BLOCK_SIZE, Int32}(ntuple(_ -> zero(Int32), BLOCK_SIZE))
    n_rows = size(neighbor_blocks, 1)
    has_neighbor = MVector{BLOCK_SIZE, Bool}(undef)

    for offset in offsets
        neighbor_cell = cell + offset
        SAFE && checkbounds(cells, neighbor_cell)
        n_points_neighbor_cell = @inbounds n_points_in_cell(cells, neighbor_cell)
        first_point_neighbor_cell = @inbounds first_entry_in_cell(cells, neighbor_cell)
        last_point_neighbor_cell = first_point_neighbor_cell + n_points_neighbor_cell - 1

        for neighbor_first_point in
            first_point_neighbor_cell:BLOCK_SIZE:last_point_neighbor_cell

            n_points_neighbor_block = min(BLOCK_SIZE,
                                          last_point_neighbor_cell - neighbor_first_point +
                                          1)

            # Full blocks are checked separately, which allows the compiler to remove
            # the repetition of the last point for the final block of a cell
            if n_points_neighbor_block == BLOCK_SIZE
                check_neighbor_block!(has_neighbor, block_coords, neighbor_coords,
                                      neighborhood_search, neighbor_first_point,
                                      BLOCK_SIZE, search_radius2)
            else
                check_neighbor_block!(has_neighbor, block_coords, neighbor_coords,
                                      neighborhood_search, neighbor_first_point,
                                      n_points_neighbor_block, search_radius2)
            end

            # Branch-free compaction: always write the block, but only advance the count
            # of a lane when the block contains a neighbor of this lane.
            # Otherwise, the block is overwritten by the next block.
            block = NeighborBlock(neighbor_first_point, n_points_neighbor_block % Int32)
            @loopinfo vectorize vectorwidth=8 for lane in 1:BLOCK_SIZE
                @inbounds begin
                    # Use a linear index with the number of rows from above, because
                    # the compiler would otherwise reload it after each store.
                    neighbor_blocks[(lane - 1) * n_rows + n_blocks[lane] + 1] = block
                    n_blocks[lane] += has_neighbor[lane]
                end
            end
        end
    end

    return SVector(n_blocks)
end

# Check for all lanes of a block at once if any point of the neighbor block of
# `n_points_neighbor_block` points starting at `neighbor_first_point` is a neighbor.
# The inner loop is fully unrolled, so that `lane_has_neighbor` can be kept
# in a SIMD register.
@inline function check_neighbor_block!(has_neighbor, block_coords, neighbor_coords,
                                       neighborhood_search, neighbor_first_point,
                                       n_points_neighbor_block, search_radius2)
    NDIMS = size(block_coords, 2)
    ELTYPE = eltype(neighborhood_search)

    @loopinfo vectorize vectorwidth=8 for lane in 1:BLOCK_SIZE
        lane_has_neighbor = false
        @loopinfo unrollfull for neighbor_lane in 1:BLOCK_SIZE
            # Unused lanes repeat the last point of the neighbor block
            neighbor = neighbor_first_point + min(neighbor_lane, n_points_neighbor_block) -
                       1

            # All points have been bounds checked in `foreach_point_blocked_inner`.
            neighbor_point_coords = @inbounds extract_svector(neighbor_coords, Val(NDIMS),
                                                              neighbor)

            # Like in `mapreduce_neighbor`, convert to the type of the search radius
            # after computing the difference
            distance2 = zero(ELTYPE)
            for dim in 1:NDIMS
                pos_diff = convert(ELTYPE,
                                   @inbounds(block_coords[lane, dim] -
                                             neighbor_point_coords[dim]))
                distance2 = muladd(pos_diff, pos_diff, distance2)
            end
            lane_has_neighbor |= distance2 <= search_radius2
        end
        @inbounds has_neighbor[lane] = lane_has_neighbor
    end

    return has_neighbor
end

# The distances in the filter are computed differently than in `mapreduce_neighbor`
# (for example with FMAs), so they might differ by rounding errors.
# To not miss any neighbors, we use a slightly larger search radius in the filter.
# `mapreduce_neighbor` then checks the exact search radius.
@inline function filter_radius2(neighborhood_search)
    ELTYPE = eltype(neighborhood_search)

    return search_radius(neighborhood_search)^2 * (1 + 16 * eps(ELTYPE))
end

@inline function mapreduce_neighbor(f, op, neighbors::SortedNeighborBlocksOfPoint; init)
    (; neighbor_coords, neighborhood_search, point, point_coords,
     neighbor_blocks, lane, n_blocks) = neighbors
    NDIMS = ndims(neighborhood_search)
    search_radius2 = search_radius(neighborhood_search)^2

    # The SIMD loads of the final blocks are shifted to the left (see below)
    last_first_point = size(neighbor_coords, 2) - BLOCK_SIZE + 1

    # Keep one partial result per SIMD lane across the complete neighbor traversal.
    # This allows LLVM to keep the partial results in vector registers and to perform the
    # reductions over the lanes only once per point instead of once per neighbor block.
    # The partial results are stored with the lanes as contiguous dimension
    # (`lane_results[lane, component]`), so that each component of the result is a vector.
    init_flat = flatten(init)
    ELTYPE = eltype(promote(init_flat...))
    lane_results = MMatrix{BLOCK_SIZE, length(init_flat), ELTYPE}(undef)
    for neighbor_lane in 1:BLOCK_SIZE
        flatten!(lane_results, neighbor_lane, init)
    end

    for block_index in 1:n_blocks
        # All points in these blocks have been bounds checked
        # in `foreach_point_blocked_inner`.
        block = @inbounds neighbor_blocks[block_index, lane]

        # Shift the lanes to the left if they would go beyond the last point.
        # Then, the lanes are always contiguous in memory, which allows SIMD loads.
        first_lane_point = min(block.first_point, last_first_point)

        # Compute all points of the block at once with SIMD instructions.
        # Each iteration only writes to its own row of `lane_results`.
        # Always computing `BLOCK_SIZE` lanes allows LLVM to keep `lane_results` in
        # registers. Lanes outside of the block are discarded.
        @ivdep @loopinfo vectorize vectorwidth=8 for neighbor_lane in 1:BLOCK_SIZE
            neighbor = first_lane_point + neighbor_lane - 1
            in_block = block.first_point <= neighbor < block.first_point + block.n_points
            neighbor_point_coords = @inbounds extract_svector(neighbor_coords, Val(NDIMS),
                                                              neighbor)

            pos_diff = convert.(eltype(neighborhood_search),
                                point_coords - neighbor_point_coords)
            distance2 = dot(pos_diff, pos_diff)
            is_neighbor = in_block & (distance2 <= search_radius2)

            # Use a valid dummy distance for points that are not neighbors.
            # Their results are discarded below.
            distance = Base.FastMath.sqrt_fast(ifelse(is_neighbor, distance2,
                                                      search_radius2))

            value = @inline f(point, neighbor, pos_diff, distance)

            # Mask the result component by component. Masking the result as a whole
            # prevents the vectorization for some functions.
            lane_result = @inbounds unflatten(init, lane_results, neighbor_lane)
            updated = flatten(@inline op(lane_result, value))
            masked = map((new, old) -> ifelse(is_neighbor, new, old),
                         updated, flatten(lane_result))
            @inbounds flatten!(lane_results, neighbor_lane, masked)
        end
    end

    # Reduce the partial results of all lanes in a tree. In each step, the upper half
    # of the remaining lanes is combined with the lower half, which is a SIMD operation
    # on contiguous lanes. Reducing lane after lane would load each lane with gathers.
    reduce_lanes!(op, init, lane_results, Val(BLOCK_SIZE ÷ 2))
    reduce_lanes!(op, init, lane_results, Val(BLOCK_SIZE ÷ 4))
    reduce_lanes!(op, init, lane_results, Val(BLOCK_SIZE ÷ 8))

    return @inbounds unflatten(init, lane_results, 1)
end
