@doc raw"""
    GridNeighborhoodSearch{NDIMS}(; search_radius = 0.0, n_points = 0,
                                  periodic_box = nothing,
                                  cell_list = DictionaryCellList{NDIMS}(),
                                  update_strategy = nothing)

Simple grid-based neighborhood search with uniform search radius.
The domain is divided into a regular grid.
For each (non-empty) grid cell, a list of points in this cell is stored.
Instead of representing a finite domain by an array of cells, a potentially infinite domain
is represented by storing cell lists in a hash table (using Julia's `Dict` data structure),
indexed by the cell index tuple
```math
\left( \left\lfloor \frac{x}{d} \right\rfloor, \left\lfloor \frac{y}{d} \right\rfloor \right) \quad \text{or} \quad
\left( \left\lfloor \frac{x}{d} \right\rfloor, \left\lfloor \frac{y}{d} \right\rfloor, \left\lfloor \frac{z}{d} \right\rfloor \right),
```
where ``x, y, z`` are the space coordinates and ``d`` is the search radius.

To find points within the search radius around a position, only points in the neighboring
cells are considered.

See also (Chalela et al., 2021), (Ihmsen et al. 2011, Section 4.4).

As opposed to (Ihmsen et al. 2011), we do not sort the points in any way,
since not sorting makes our implementation a lot faster (although less parallelizable).

# Arguments
- `NDIMS`: Number of dimensions.

# Keywords
- `search_radius = 0.0`:    The fixed search radius. The default of `0.0` is useful together
                            with [`copy_neighborhood_search`](@ref).
- `n_points = 0`:           Total number of points. The default of `0` is useful together
                            with [`copy_neighborhood_search`](@ref).
- `periodic_box = nothing`: In order to use a (rectangular) periodic domain, pass a
                            [`PeriodicBox`](@ref).
- `cell_list`:              The cell list that maps a cell index to a list of points inside
                            the cell. By default, a [`DictionaryCellList`](@ref) is used.
- `update_strategy = nothing`: Strategy to parallelize `update!`. Available options are:
    - `nothing`: Automatically choose the best available option.
    - [`ParallelUpdate()`](@ref): This is not available for all cell list implementations.
    - [`SemiParallelUpdate()`](@ref): This is available for all cell list implementations
        and is the default when available.
    - [`SerialIncrementalUpdate()`](@ref)
    - [`SerialUpdate()`](@ref)

## References
- M. Chalela, E. Sillero, L. Pereyra, M.A. Garcia, J.B. Cabral, M. Lares, M. Merchán.
  "GriSPy: A Python package for fixed-radius nearest neighbors search".
  In: Astronomy and Computing 34 (2021).
  [doi: 10.1016/j.ascom.2020.100443](https://doi.org/10.1016/j.ascom.2020.100443)
- Markus Ihmsen, Nadir Akinci, Markus Becker, Matthias Teschner.
  "A Parallel SPH Implementation on Multi-Core CPUs".
  In: Computer Graphics Forum 30.1 (2011), pages 99–112.
  [doi: 10.1111/J.1467-8659.2010.01832.X](https://doi.org/10.1111/J.1467-8659.2010.01832.X)
"""
struct GridNeighborhoodSearch{NDIMS, US, CL, ELTYPE, PB, UB} <: AbstractNeighborhoodSearch
    cell_list       :: CL
    search_radius   :: ELTYPE
    periodic_box    :: PB
    n_cells         :: NTuple{NDIMS, Int}    # Required to calculate periodic cell index
    cell_size       :: NTuple{NDIMS, ELTYPE} # Required to calculate cell index
    update_buffer   :: UB                    # Multithreaded buffer for `update!`
    update_strategy :: US
end

function GridNeighborhoodSearch{NDIMS}(; search_radius = 0.0, n_points = 0,
                                       periodic_box = nothing,
                                       cell_list = DictionaryCellList{NDIMS}(),
                                       update_strategy = nothing) where {NDIMS}
    if isnothing(update_strategy)
        # Automatically choose best available update option for this cell list
        update_strategy = first(supported_update_strategies(cell_list))()
    elseif !(typeof(update_strategy) in supported_update_strategies(cell_list))
        throw(ArgumentError("$update_strategy is not a valid update strategy for " *
                            "this cell list. Available options are " *
                            "$(supported_update_strategies(cell_list))"))
    end

    update_buffer = create_update_buffer(update_strategy, cell_list, n_points)

    if search_radius < eps() || isnothing(periodic_box)
        # No periodicity
        n_cells = ntuple(_ -> -1, Val(NDIMS))
        cell_size = ntuple(_ -> search_radius, Val(NDIMS))
    else
        # Round up search radius so that the grid fits exactly into the domain without
        # splitting any cells. This might impact performance slightly, since larger
        # cells mean that more potential neighbors are considered than necessary.
        # Allow small tolerance to avoid inefficient larger cells due to machine
        # rounding errors.
        n_cells = Tuple(floor.(Int, (periodic_box.size .+ 10eps()) / search_radius))
        cell_size = Tuple(periodic_box.size ./ n_cells)

        if any(i -> i < 3, n_cells)
            throw(ArgumentError("the `GridNeighborhoodSearch` needs at least 3 cells " *
                                "in each dimension when used with periodicity. " *
                                "Please use no NHS for very small problems."))
        end
    end

    return GridNeighborhoodSearch(cell_list, search_radius, periodic_box, n_cells,
                                  cell_size, update_buffer, update_strategy)
end

@inline Base.ndims(::GridNeighborhoodSearch{NDIMS}) where {NDIMS} = NDIMS

@inline requires_update(::GridNeighborhoodSearch) = (false, true)

"""
    ParallelUpdate()

Fully parallel initialization and update by using atomic operations to avoid race conditions
when adding points into the same cell.
This is not available for all cell list implementations.

See [`GridNeighborhoodSearch`](@ref) for usage information.
"""
struct ParallelUpdate end

"""
    ParallelIncrementalUpdate()

Like [`ParallelUpdate`](@ref), but only updates the cells that have changed.
This is generally slower than a full reinitialization with [`ParallelUpdate`](@ref),
but is included for benchmarking purposes.
This is not available for all cell list implementations, but is the default when available.

See [`GridNeighborhoodSearch`](@ref) for usage information.
"""
struct ParallelIncrementalUpdate end

"""
    SemiParallelUpdate()

Loop over all cells in parallel to mark cells with points that now belong to a different
cell. Then, move points of affected cells serially to avoid race conditions.
This is available for all cell list implementations and is the default when
[`ParallelUpdate`](@ref) is not available.

See [`GridNeighborhoodSearch`](@ref) for usage information.
"""
struct SemiParallelUpdate end

"""
    SerialIncrementalUpdate()

Deactivate parallelization in the neighborhood search update.
Parallel neighborhood search update can be one of the largest sources of error variations
between simulations with different thread numbers due to neighbor ordering changes.
This strategy incrementally updates the cell lists in every update.

See [`GridNeighborhoodSearch`](@ref) for usage information.
"""
struct SerialIncrementalUpdate end

"""
    SerialUpdate()

Deactivate parallelization in the neighborhood search update.
Parallel neighborhood search update can be one of the largest sources of error variations
between simulations with different thread numbers due to neighbor ordering changes.
This strategy reinitializes the cell lists in every update.

See [`GridNeighborhoodSearch`](@ref) for usage information.
"""
struct SerialUpdate end

# No update buffer needed for non-incremental update/initialize
@inline function create_update_buffer(::Union{SerialUpdate, ParallelUpdate}, _, _)
    return nothing
end

@inline function create_update_buffer(::ParallelIncrementalUpdate, cell_list, _)
    # Create empty `lengths` vector to read from while writing to `cell_list.cells.lengths`
    n_cells = length(each_cell_index(cell_list))
    return Vector{Int32}(undef, n_cells)
end

@inline function create_update_buffer(::SemiParallelUpdate, cell_list, n_points)
    # Create update buffer and initialize it with empty vectors
    update_buffer = DynamicVectorOfVectors{index_type(cell_list)}(max_outer_length = Threads.nthreads(),
                                                                  max_inner_length = n_points)
    push!(update_buffer, (index_type(cell_list)[] for _ in 1:Threads.nthreads())...)
end

@inline function create_update_buffer(::SerialIncrementalUpdate, cell_list, n_points)
    # Create update buffer and initialize it with empty vectors.
    # Only one thread is used here, so we only need one element in the buffer.
    update_buffer = DynamicVectorOfVectors{index_type(cell_list)}(max_outer_length = 1,
                                                                  max_inner_length = n_points)
    push!(update_buffer, index_type(cell_list)[])
end

function initialize!(neighborhood_search::GridNeighborhoodSearch,
                     x::AbstractMatrix, y::AbstractMatrix;
                     parallelization_backend = default_backend(x),
                     eachindex_y = axes(y, 2))
    initialize_grid!(neighborhood_search, y; parallelization_backend, eachindex_y)
end

function initialize_grid!(neighborhood_search::GridNeighborhoodSearch, y::AbstractMatrix;
                          parallelization_backend = default_backend(y),
                          eachindex_y = axes(y, 2))
    (; cell_list) = neighborhood_search

    empty!(cell_list)

    if neighborhood_search.search_radius < eps()
        # Cannot initialize with zero search radius.
        # This is used in TrixiParticles when a neighborhood search is not used.
        return neighborhood_search
    end

    @boundscheck checkbounds(y, eachindex_y)

    # Ignore the parallelization backend here. This cannot be parallelized.
    for point in eachindex_y
        # Get cell index of the point's cell
        point_coords = @inbounds extract_svector(y, Val(ndims(neighborhood_search)), point)
        cell = cell_coords(point_coords, neighborhood_search)

        # Add point to corresponding cell
        push_cell!(cell_list, cell, point)
    end

    return neighborhood_search
end

function initialize_grid!(neighborhood_search::GridNeighborhoodSearch{<:Any,
                                                                      ParallelUpdate},
                          y::AbstractMatrix; parallelization_backend = default_backend(y),
                          eachindex_y = axes(y, 2))
    (; cell_list) = neighborhood_search

    empty!(cell_list)

    if neighborhood_search.search_radius < eps()
        # Cannot initialize with zero search radius.
        # This is used in TrixiParticles when a neighborhood search is not used.
        return neighborhood_search
    end

    @boundscheck checkbounds(y, eachindex_y)

    @threaded parallelization_backend for point in eachindex_y
        # Get cell index of the point's cell
        point_coords = @inbounds extract_svector(y, Val(ndims(neighborhood_search)), point)
        cell = cell_coords(point_coords, neighborhood_search)

        # Add point to corresponding cell
        push_cell_atomic!(cell_list, cell, point)
    end

    return neighborhood_search
end

function update!(neighborhood_search::GridNeighborhoodSearch,
                 x::AbstractMatrix, y::AbstractMatrix;
                 points_moving = (true, true), parallelization_backend = default_backend(x),
                 eachindex_y = axes(y, 2))
    # The coordinates of the first set of points are irrelevant for this NHS.
    # Only update when the second set is moving.
    points_moving[2] || return neighborhood_search

    update_grid!(neighborhood_search, y; eachindex_y, parallelization_backend)
end

# Update only with neighbor coordinates
function update_grid!(neighborhood_search::Union{GridNeighborhoodSearch{<:Any,
                                                                        SerialIncrementalUpdate},
                                                 GridNeighborhoodSearch{<:Any,
                                                                        SemiParallelUpdate}},
                      y::AbstractMatrix;
                      parallelization_backend = default_backend(y),
                      eachindex_y = axes(y, 2))
    (; cell_list, update_buffer) = neighborhood_search

    if eachindex_y != axes(y, 2)
        # Incremental update doesn't support inactive points
        error("this neighborhood search/update strategy does not support inactive points")
    end

    # Empty each thread's list
    @threaded parallelization_backend for i in eachindex(update_buffer)
        emptyat!(update_buffer, i)
    end

    # Find all cells containing points that now belong to another cell.
    # This loop is threaded for `update_strategy == SemiParallelUpdate()`.
    mark_changed_cells!(neighborhood_search, y, parallelization_backend)

    # Iterate over all marked cells and move the points into their new cells.
    # This is always a serial loop (hence "semi-parallel").
    for j in eachindex(update_buffer)
        for cell_index in update_buffer[j]
            points = cell_list[cell_index]

            # Find all points whose coordinates do not match this cell.
            #
            # WARNING!!!
            # The `DynamicVectorOfVectors` requires this loop to be **in descending order**.
            # `deleteat_cell!(..., i)` will change the order of points that come after `i`.
            for i in reverse(eachindex(points))
                point = points[i]
                point_coords = extract_svector(y, Val(ndims(neighborhood_search)), point)
                new_cell_coords = cell_coords(point_coords, neighborhood_search)

                if !is_correct_cell(cell_list, new_cell_coords, cell_index)
                    # Add point to new cell or create cell if it does not exist
                    push_cell!(cell_list, new_cell_coords, point)

                    # Remove moved point from this cell
                    deleteat_cell!(cell_list, cell_index, i)
                end
            end
        end
    end

    return neighborhood_search
end

@inline function mark_changed_cells!(neighborhood_search::GridNeighborhoodSearch{<:Any,
                                                                                 SemiParallelUpdate},
                                     y, parallelization_backend)
    (; cell_list, update_buffer) = neighborhood_search

    # `each_cell_index(cell_list)` might return a `KeySet`, which has to be `collect`ed
    # first to support indexing.
    eachcell = each_cell_index_threadable(cell_list)

    # Use chunks (usually one per thread) to index into the update buffer.
    # We cannot use `Iterators.partition` here, since the resulting iterator does not
    # support indexing and therefore cannot be used in a threaded loop.
    chunk_length = div(length(eachcell), length(update_buffer), RoundUp)

    @threaded parallelization_backend for chunk_id in 1:length(update_buffer)
        # Manual partitioning of `eachcell`
        start = (chunk_length * (chunk_id - 1)) + 1
        end_ = min(chunk_length * chunk_id, length(eachcell))

        for i in start:end_
            cell_index = eachcell[i]

            mark_changed_cell!(neighborhood_search, cell_index, y, chunk_id)
        end
    end
end

@inline function mark_changed_cells!(neighborhood_search::GridNeighborhoodSearch{<:Any,
                                                                                 SerialIncrementalUpdate},
                                     y, _)
    (; cell_list) = neighborhood_search

    # Ignore the parallelization backend here for `SerialIncrementalUpdate`.
    for cell_index in each_cell_index(cell_list)
        # `chunk_id` is always `1` for `SerialIncrementalUpdate`
        mark_changed_cell!(neighborhood_search, cell_index, y, 1)
    end
end

@inline function mark_changed_cell!(neighborhood_search, cell_index, y, chunk_id)
    (; cell_list, update_buffer) = neighborhood_search

    for point in cell_list[cell_index]
        point_coords = extract_svector(y, Val(ndims(neighborhood_search)), point)
        cell = cell_coords(point_coords, neighborhood_search)

        # `cell` is a tuple, `cell_index` is the linear index used internally by the
        # cell list to store cells inside `cell`.
        # These can be identical (see `DictionaryCellList`).
        if !is_correct_cell(cell_list, cell, cell_index)
            # Mark this cell and continue with the next one
            pushat!(update_buffer, chunk_id, cell_index)
            break
        end
    end
end

# Fully parallel incremental update with atomic push.
function update_grid!(neighborhood_search::GridNeighborhoodSearch{<:Any,
                                                                  ParallelIncrementalUpdate},
                      y::AbstractMatrix; parallelization_backend = default_backend(y),
                      eachindex_y = axes(y, 2))
    (; cell_list, update_buffer) = neighborhood_search

    if eachindex_y != axes(y, 2)
        # Incremental update doesn't support inactive points
        error("this neighborhood search/update strategy does not support inactive points")
    end

    # Note that we need two separate loops for adding and removing points.
    # `push_cell_atomic!` only guarantees thread-safety when different threads push
    # simultaneously, but it does not work when `deleteat_cell!` is called at the same time.

    # While pushing to the cell list, iterating over the cell lists is not safe.
    # We can work around this by using the old lengths.
    # TODO this is hardcoded for the `FullGridCellList`
    @threaded parallelization_backend for i in eachindex(update_buffer,
                                                    cell_list.cells.lengths)
        update_buffer[i] = cell_list.cells.lengths[i]
    end

    # Add points to new cells
    @threaded parallelization_backend for cell_index in
                                          each_cell_index_threadable(cell_list)
        for i in 1:update_buffer[cell_index]
            point = cell_list.cells.backend[i, cell_index]
            point_coords = extract_svector(y, Val(ndims(neighborhood_search)), point)
            new_cell_coords = cell_coords(point_coords, neighborhood_search)

            if !is_correct_cell(cell_list, new_cell_coords, cell_index)
                # Add point to new cell or create cell if it does not exist
                push_cell_atomic!(cell_list, new_cell_coords, point)
            end
        end
    end

    # Remove points from old cells
    @threaded parallelization_backend for cell_index in
                                          each_cell_index_threadable(cell_list)
        points = cell_list[cell_index]

        # WARNING!!!
        # The `DynamicVectorOfVectors` requires this loop to be **in descending order**.
        # `deleteat_cell!(..., i)` will change the order of points that come after `i`.
        for i in reverse(eachindex(points))
            point = points[i]
            point_coords = extract_svector(y, Val(ndims(neighborhood_search)), point)
            new_cell_coords = cell_coords(point_coords, neighborhood_search)

            if !is_correct_cell(cell_list, new_cell_coords, cell_index)
                # Remove moved point from this cell
                deleteat_cell!(cell_list, cell_index, i)
            end
        end
    end

    return neighborhood_search
end

# Non-incremental update strategies just forward to `initialize_grid!`
function update_grid!(neighborhood_search::Union{GridNeighborhoodSearch{<:Any,
                                                                        ParallelUpdate},
                                                 GridNeighborhoodSearch{<:Any,
                                                                        SerialUpdate}},
                      y::AbstractMatrix; parallelization_backend = default_backend(y),
                      eachindex_y = axes(y, 2))
    initialize_grid!(neighborhood_search, y; parallelization_backend, eachindex_y)
end

function check_collision(neighbor_cell_, neighbor_coords, cell_list, nhs)
    # This is only relevant for the `SpatialHashingCellList`
    return false
end

# Check if `neighbor_coords` belong to `neighbor_cell`, which might not be the case
# with the `SpatialHashingCellList` if this cell has a collision.
function check_collision(neighbor_cell_::CartesianIndex, neighbor_coords,
                         cell_list::SpatialHashingCellList, nhs)
    (; list_size, collisions, coords) = cell_list
    neighbor_cell = periodic_cell_index(Tuple(neighbor_cell_), nhs)

    return neighbor_cell != cell_coords(neighbor_coords, nhs)
end

function check_cell_collision(neighbor_cell_::CartesianIndex,
                              cell_list, nhs)
    # This is only relevant for the `SpatialHashingCellList`
    return false
end

# Check if there is a collision in this cell, meaning there is at least one point
# in this list that doesn't actually belong in this cell.
function check_cell_collision(neighbor_cell_::CartesianIndex,
                              cell_list::SpatialHashingCellList, nhs)
    (; list_size, collisions, coords) = cell_list
    neighbor_cell = periodic_cell_index(Tuple(neighbor_cell_), nhs)
    hash = spatial_hash(neighbor_cell, list_size)

    # `collisions[hash] == true` means points from multiple cells are in this list.
    # `collisions[hash] == false` means points from only one cells are in this list.
    # We could still have a collision though, if this one cell is not `neighbor_cell`,
    # which is possible when `neighbor_cell` is empty.
    return collisions[hash] || coords[hash] != neighbor_cell
end

# Specialized version of the function in `neighborhood_search.jl`, which is faster
# than looping over `eachneighbor`.
@inline function foreach_neighbor(f, neighbor_system_coords,
                                  neighborhood_search::GridNeighborhoodSearch,
                                  point, point_coords, search_radius)
    (; cell_list, periodic_box) = neighborhood_search
    cell = cell_coords(point_coords, neighborhood_search)

    for neighbor_cell_ in neighboring_cells(cell, neighborhood_search)
        neighbor_cell = Tuple(neighbor_cell_)
        neighbors = points_in_cell(neighbor_cell, neighborhood_search)

        # Boolean to indicate if this cell has a collision (only with `SpatialHashingCellList`)
        cell_collision = check_cell_collision(neighbor_cell_,
                                              cell_list, neighborhood_search)

        for neighbor_ in eachindex(neighbors)
            neighbor = @inbounds neighbors[neighbor_]

            # Making the following `@inbounds` yields a ~2% speedup on an NVIDIA H100.
            # But we don't know if `neighbor` (extracted from the cell list) is in bounds.
            neighbor_coords = extract_svector(neighbor_system_coords,
                                              Val(ndims(neighborhood_search)), neighbor)

            pos_diff = point_coords - neighbor_coords
            distance2 = dot(pos_diff, pos_diff)

            pos_diff,
            distance2 = compute_periodic_distance(pos_diff, distance2,
                                                  search_radius, periodic_box)

            if distance2 <= search_radius^2
                distance = sqrt(distance2)

                # If this cell has a collision, check if this point belongs to this cell
                # (only with `SpatialHashingCellList`).
                if cell_collision &&
                   check_collision(neighbor_cell_, neighbor_coords, cell_list,
                                   neighborhood_search)
                    continue
                end

                # Inline to avoid loss of performance
                # compared to not using `foreach_point_neighbor`.
                @inline f(point, neighbor, pos_diff, distance)
            end
        end
    end
end

@inline function neighboring_cells(cell, neighborhood_search)
    NDIMS = ndims(neighborhood_search)

    # For `cell = (x, y, z)`, this returns Cartesian indices
    # {x-1, x, x+1} × {y-1, y, y+1} × {z-1, z, z+1}.
    return CartesianIndices(ntuple(i -> (cell[i] - 1):(cell[i] + 1), NDIMS))
end

@inline function eachneighbor(coords, neighborhood_search::GridNeighborhoodSearch)
    cell = cell_coords(coords, neighborhood_search)

    # Merge all lists of points in the neighboring cells into one iterator
    Iterators.flatten(points_in_cell(Tuple(cell), neighborhood_search)
                      for cell in neighboring_cells(cell, neighborhood_search))
end

@propagate_inbounds function points_in_cell(cell_index, neighborhood_search)
    (; cell_list) = neighborhood_search

    return cell_list[periodic_cell_index(cell_index, neighborhood_search)]
end

@inline function periodic_cell_index(cell_index, neighborhood_search)
    (; n_cells, periodic_box, cell_list) = neighborhood_search

    periodic_cell_index(cell_index, periodic_box, n_cells, cell_list)
end

@inline periodic_cell_index(cell_index, ::Nothing, n_cells, cell_list) = cell_index

@inline function periodic_cell_index(cell_index, ::PeriodicBox, n_cells, cell_list)
    # 1-based modulo
    return mod1.(cell_index, n_cells)
end

@inline function cell_coords(coords, neighborhood_search)
    (; periodic_box, cell_list, cell_size) = neighborhood_search

    return cell_coords(coords, periodic_box, cell_list, cell_size)
end

@inline function cell_coords(coords, periodic_box::Nothing, cell_list, cell_size)
    return Tuple(floor_to_int.(coords ./ cell_size))
end

@inline function cell_coords(coords, periodic_box::PeriodicBox, cell_list, cell_size)
    # Subtract `min_corner` to offset coordinates so that the min corner of the periodic
    # box corresponds to the (0, 0, 0) cell of the NHS.
    # This way, there are no partial cells in the domain if the domain size is an integer
    # multiple of the cell size (which is required, see the constructor).
    offset_coords = periodic_coords(coords, periodic_box) .- periodic_box.min_corner

    # Add one for 1-based indexing. The min corner will be the (1, 1, 1)-cell.
    return Tuple(floor_to_int.(offset_coords ./ cell_size)) .+ 1
end

function copy_neighborhood_search(nhs::GridNeighborhoodSearch, search_radius, n_points;
                                  eachpoint = 1:n_points)
    (; periodic_box) = nhs

    cell_list = copy_cell_list(nhs.cell_list, search_radius, periodic_box)

    return GridNeighborhoodSearch{ndims(nhs)}(; search_radius, n_points, periodic_box,
                                              cell_list,
                                              update_strategy = nhs.update_strategy)
end
