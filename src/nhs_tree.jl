@doc raw"""
    TreeNeighborhoodSearch{NDIMS}(; cell_list, n_points = 0, update_strategy = nothing)

Tree-based neighborhood search with a variable search radius ``h_j`` for each point ``j``.
For a query point ``x`` with search radius ``h``, all points ``j`` with
```math
\| x - y_j \| \leq \frac{h + h_j}{2}
```
are considered neighbors.

The tree is constructed with [`initialize_tree!`](@ref) or [`update_tree!`](@ref)
from the neighbor coordinates and their search radii.
Neighbors can then be queried for arbitrary points with [`foreach_neighbor`](@ref)
or [`mapreduce_neighbor`](@ref), where the keyword argument `search_radius`
is the search radius ``h`` of the query point.
Since the search radius of each query point is required, [`foreach_point_neighbor`](@ref)
is not supported.

The points are stored in the leaves of a quadtree (2D) or octree (3D),
see [`TreeCellList`](@ref).
The leaf size is chosen locally based on the search radii of the points,
without any assumptions on the variation of the search radii.
As with the [`GridNeighborhoodSearch`](@ref), a query only considers the ``3^d`` block of
cells around the leaf containing the query point, at the level of this leaf.
Leaves in this block are skipped when their distance to the query point is larger than
the average of the query radius and the maximum search radius of all points in the leaf.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in any future releases.

# Arguments
- `NDIMS`: Number of dimensions.

# Keywords
- `cell_list`:       A [`TreeCellList`](@ref) defining the domain and the depth of the tree.
- `n_points = 0`:    Total number of points in the neighbor coordinates array.
                     The default of `0` is useful together with
                     [`copy_neighborhood_search`](@ref).
- `update_strategy = nothing`: Strategy to parallelize the update. The tree is always
                     rebuilt from scratch. Available options are:
    - `nothing`: Automatically choose the best available option.
    - [`SerialUpdate()`](@ref)

# Correctness
Denote by ``s_L`` the size of the cells at level ``L``.
A cell at level ``L`` is only split if ``s_{L+1} \geq h_j`` for all points ``j``
in the ``3^d`` block of cells around this cell at level ``L``.
Consider a query point ``x`` with search radius ``h`` in a leaf at level ``L_x``
and let ``q \leq L_x`` be the finest level with ``s_q \geq h``.
We search for neighbors in the ``3^d`` block around the cell at level ``q`` containing ``x``.
Now, consider a point ``j`` outside of this block.
Then, let ``m < q`` be the finest level such that the ``3^d`` block around the cell at level
``m`` containing ``x`` contains ``j``. Since this cell has been split, we have
``s_{m+1} \geq h_j``. Since ``j`` is not in the ``3^d`` block at level ``m + 1``,
we have ``\| x - y_j \| > s_{m+1} \geq (h + h_j) / 2``.
Therefore, ``j`` is not a neighbor of ``x``.

For a point ``x = y_i`` of the tree with ``h = h_i``, the leaf containing ``x`` satisfies
``s_{L_x} \geq h``, so we always search at the level of the leaf, like with the
[`GridNeighborhoodSearch`](@ref).
"""
struct TreeNeighborhoodSearch{NDIMS, US, CL, R} <: AbstractNeighborhoodSearch
    cell_list       :: CL
    search_radii    :: R  # Copy of the search radii of the neighbor points
    update_strategy :: US
end

function TreeNeighborhoodSearch{NDIMS}(; cell_list, n_points = 0,
                                       update_strategy = nothing) where {NDIMS}
    if !(cell_list isa TreeCellList)
        throw(ArgumentError("`TreeNeighborhoodSearch` requires a `TreeCellList`"))
    end

    if ndims(cell_list) != NDIMS
        throw(ArgumentError("a $(NDIMS)D cell list is required for " *
                            "a TreeNeighborhoodSearch{$(NDIMS)}"))
    end

    if isnothing(update_strategy)
        # Automatically choose best available update option for this cell list
        update_strategy = first(supported_update_strategies(cell_list))()
    elseif !(typeof(update_strategy) in supported_update_strategies(cell_list))
        throw(ArgumentError("$update_strategy is not a valid update strategy for " *
                            "this cell list. Available options are " *
                            "$(supported_update_strategies(cell_list))"))
    end

    search_radii = zeros(eltype(cell_list), n_points)

    return TreeNeighborhoodSearch{NDIMS, typeof(update_strategy), typeof(cell_list),
                                  typeof(search_radii)}(cell_list, search_radii,
                                                        update_strategy)
end

@inline Base.ndims(::TreeNeighborhoodSearch{NDIMS}) where {NDIMS} = NDIMS
@inline Base.eltype(neighborhood_search::TreeNeighborhoodSearch) = eltype(neighborhood_search.cell_list)

@inline requires_update(::TreeNeighborhoodSearch) = (false, true)

@inline function search_radius(::TreeNeighborhoodSearch)
    # This is the default value of the keyword argument `search_radius`
    # of `foreach_neighbor` and `mapreduce_neighbor`.
    error("`TreeNeighborhoodSearch` requires the search radius of the query point " *
          "to be passed as keyword argument `search_radius`")
end

function foreach_point_neighbor(f, system_coords, neighbor_coords,
                                neighborhood_search::TreeNeighborhoodSearch; kwargs...)
    error("`TreeNeighborhoodSearch` requires the search radius of the query point " *
          "to be passed as keyword argument `search_radius`, so `foreach_point_neighbor` " *
          "is not supported. Use `foreach_neighbor` instead.")
end

# Every query radius is supported
@inline check_search_radius(::TreeNeighborhoodSearch, _) = nothing

function initialize!(::TreeNeighborhoodSearch, x::AbstractMatrix, y::AbstractMatrix;
                     kwargs...)
    error("`TreeNeighborhoodSearch` requires search radii. Use `initialize_tree!` instead.")
end

function update!(::TreeNeighborhoodSearch, x::AbstractMatrix, y::AbstractMatrix;
                 kwargs...)
    error("`TreeNeighborhoodSearch` requires search radii. Use `update_tree!` instead.")
end

"""
    initialize_tree!(search::TreeNeighborhoodSearch, y, search_radii;
                     parallelization_backend = default_backend(y),
                     eachindex_y = axes(y, 2))

Build the tree of a [`TreeNeighborhoodSearch`](@ref) for the neighbor coordinates `y`
with search radii `search_radii`.
`y` is expected to be a matrix, where the `j`-th column contains the coordinates of point `j`
with the search radius `search_radii[j]`.
The search radii are copied, so `search_radii` can be modified afterwards.

Optionally, when points in `y` are to be ignored, the keyword argument `eachindex_y` can be
passed to specify the indices of the points in `y` that are to be used.

See also [`update_tree!`](@ref).
"""
function initialize_tree!(neighborhood_search::TreeNeighborhoodSearch, y::AbstractMatrix,
                          search_radii::AbstractVector;
                          parallelization_backend = default_backend(y),
                          eachindex_y = axes(y, 2))
    (; cell_list) = neighborhood_search

    if length(search_radii) != size(y, 2)
        throw(DimensionMismatch("the number of search radii must match the number " *
                                "of points in `y`"))
    end
    @boundscheck checkbounds(y, ndims(neighborhood_search), eachindex_y)

    resize!(neighborhood_search.search_radii, length(search_radii))
    copyto!(neighborhood_search.search_radii, search_radii)

    rebuild!(cell_list, y, neighborhood_search.search_radii, eachindex_y)

    return neighborhood_search
end

"""
    update_tree!(search::TreeNeighborhoodSearch, y, search_radii;
                 parallelization_backend = default_backend(y),
                 eachindex_y = axes(y, 2))

Update the tree of a [`TreeNeighborhoodSearch`](@ref) for the neighbor coordinates `y`
with search radii `search_radii`.
The tree is always rebuilt from scratch, so this is equivalent to [`initialize_tree!`](@ref).
"""
function update_tree!(neighborhood_search::TreeNeighborhoodSearch, y::AbstractMatrix,
                      search_radii::AbstractVector;
                      parallelization_backend = default_backend(y),
                      eachindex_y = axes(y, 2))
    initialize_tree!(neighborhood_search, y, search_radii;
                     parallelization_backend, eachindex_y)
end

# The search radius of the pair of a query point with search radius `search_radius`
# and the neighbor point `neighbor` is the average of both search radii.
@propagate_inbounds function pair_search_radius(neighborhood_search::TreeNeighborhoodSearch,
                                                search_radius, neighbor)
    return (search_radius + neighborhood_search.search_radii[neighbor]) / 2
end

# Note that calling this function with `@inbounds` is not safe.
# See the comments in `foreach_neighbor_unsafe`.
@propagate_inbounds function mapreduce_neighbor_inner(f, op, neighbor_coords,
                                                      neighborhood_search::TreeNeighborhoodSearch,
                                                      point, point_coords,
                                                      search_radius, init)
    (; cell_list) = neighborhood_search
    (; max_level) = cell_list
    NDIMS = ndims(neighborhood_search)

    query_radius = convert(eltype(neighborhood_search), search_radius)
    finest_cell = cell_coords(point_coords, cell_list, max_level)

    if is_in_grid(finest_cell, cell_list, max_level)
        # Making the following `@inbounds` is not safe because we don't know if
        # the tree has been initialized correctly.
        leaf_level_ = leaf_level(cell_list, cell_index(cell_list, finest_cell, max_level))

        # Search at a coarser level when the query radius is larger than the leaf
        level = query_level(cell_list, leaf_level_, query_radius)
    else
        # Points outside the grid are treated as if they were in a leaf at level 0
        level = 0
    end

    block_min,
    block_max = search_block(cell_list, finest_cell, level, point_coords,
                             query_radius)

    reduced = init
    for neighbor_cell in CartesianIndices(ntuple(i -> block_min[i]:block_max[i],
                                Val(NDIMS)))
        reduced = mapreduce_cell(f, op, reduced, Tuple(neighbor_cell), level, block_min,
                                 neighbor_coords, neighborhood_search, point,
                                 point_coords, query_radius)
    end

    return reduced
end

# The first and the last cell (zero-based Cartesian coordinates) of the block of cells
# at level `level` that has to be searched for neighbors of a point.
@propagate_inbounds function search_block(cell_list, finest_cell, level, point_coords,
                                          query_radius)
    (; max_level, max_radius_all, min_corner, root_cell_size, n_roots) = cell_list

    if level > 0
        # Search the 3^d block around the cell containing the point at level `level`.
        # Compute the cell from the finest cell to be consistent with the leaf lookup.
        cell = finest_cell .>> (max_level - level)
        n_cells = n_cells_per_dimension(cell_list, level)

        return max.(cell .- 1, 0), min.(cell .+ 1, n_cells .- 1)
    end

    # At level 0, search all root cells intersecting the box around the point
    # in which all neighbors must be.
    # This is equivalent to the 3^d block of root cells when all search radii
    # are smaller than the root cells, but also handles larger search radii
    # and points outside of the grid.
    max_pair_radius = (query_radius + max_radius_all[1]) / 2
    block_min = Tuple(floor_to_int.((point_coords .- max_pair_radius .- min_corner) ./
                                    root_cell_size))
    block_max = Tuple(floor_to_int.((point_coords .+ max_pair_radius .- min_corner) ./
                                    root_cell_size))

    return max.(block_min, 0), min.(block_max, n_roots .- 1)
end

# The finest level `<= leaf_level` with cells that are at least as large as `query_radius`
@inline function query_level(cell_list, leaf_level, query_radius)
    level = leaf_level
    while level > 0 && cell_size(cell_list, level) < query_radius
        level -= 1
    end

    return level
end

# Reduce over all neighbors in the cell `cell` at level `level` of the search block,
# which starts at the cell `block_min`.
# This cell can be inside a coarser leaf, be a leaf, or be split into finer leaves.
@propagate_inbounds function mapreduce_cell(f, op, reduced, cell, level, block_min,
                                            neighbor_coords, neighborhood_search,
                                            point, point_coords, query_radius)
    (; cell_list) = neighborhood_search
    NDIMS = ndims(neighborhood_search)

    cell_index_ = cell_index(cell_list, cell, level)

    # First finest cell inside this cell
    first_finest_cell = cell_index_ << (NDIMS * (cell_list.max_level - level))
    leaf_level_ = leaf_level(cell_list, first_finest_cell)

    if leaf_level_ < level
        # This cell is inside a coarser leaf, which might contain multiple cells of the
        # search block. Only visit this leaf once, from the first cell of the block
        # inside this leaf (with the smallest coordinates in each dimension).
        level_difference = level - leaf_level_
        leaf_min = (cell .>> level_difference) .<< level_difference
        if cell != max.(leaf_min, block_min)
            return reduced
        end

        # Skip this leaf if it is too far away from the query point
        leaf_index = cell_index_ >> (NDIMS * level_difference)
        leaf_cell = cell .>> level_difference
        if is_out_of_reach(cell_list, leaf_cell, leaf_index, leaf_level_,
                           point_coords, query_radius)
            return reduced
        end

        # First finest cell of the leaf
        first_cell = leaf_index << (NDIMS * (cell_list.max_level - leaf_level_))
        last_cell = first_cell + n_finest_cells(cell_list, leaf_level_)

        return mapreduce_finest_cells(f, op, reduced, first_cell, last_cell,
                                      neighbor_coords, neighborhood_search,
                                      point, point_coords, query_radius)
    end

    # Skip this cell and all leaves inside if it is too far away from the query point
    if is_out_of_reach(cell_list, cell, cell_index_, level, point_coords, query_radius)
        return reduced
    end

    # Loop over all leaves inside this cell (or this cell if it is a leaf).
    # Leaves are contiguous ranges of finest cells, so we can jump from leaf to leaf.
    finest_cell = first_finest_cell
    end_cell = first_finest_cell + n_finest_cells(cell_list, level)
    while finest_cell < end_cell
        leaf_level_ = leaf_level(cell_list, finest_cell)
        next_cell = finest_cell + n_finest_cells(cell_list, leaf_level_)

        # Skip finer leaves that are too far away from the query point.
        # Note that a leaf at `level` is this cell itself, which has been checked above.
        level_difference = cell_list.max_level - leaf_level_
        leaf_index = finest_cell >> (NDIMS * level_difference)
        if leaf_level_ == level ||
           !is_out_of_reach(cell_list,
                            cell_coords_from_index(cell_list, leaf_index, leaf_level_),
                            leaf_index, leaf_level_, point_coords, query_radius)
            reduced = mapreduce_finest_cells(f, op, reduced, finest_cell, next_cell,
                                             neighbor_coords, neighborhood_search,
                                             point, point_coords, query_radius)
        end

        finest_cell = next_cell
    end

    return reduced
end

# Check if all points in the cell `cell` (zero-based Cartesian coordinates) with index
# `cell_index` at level `level` are too far away from the query point to be neighbors.
# This is the case when the minimum distance from the query point to the cell is larger
# than the average of the query radius and the maximum radius in this cell.
@propagate_inbounds function is_out_of_reach(cell_list, cell, cell_index, level,
                                             point_coords, query_radius)
    (; min_corner, max_radius, root_cell_size, n_roots) = cell_list
    ELTYPE = eltype(cell_list)

    # Enlarge the cell by a safety margin to account for rounding errors when assigning
    # points to cells. These are in the order of machine precision times the domain size.
    margin = 16 * eps(ELTYPE) * root_cell_size * maximum(n_roots)

    size_ = cell_size(cell_list, level)
    cell_min = min_corner .+ SVector(cell) .* size_ .- margin
    cell_max = cell_min .+ size_ .+ 2 * margin

    # Distance vector from the query point to the closest point in the cell
    distance_vector = max.(cell_min .- point_coords, point_coords .- cell_max, 0)
    distance2 = dot(distance_vector, distance_vector)

    max_pair_radius = (query_radius +
                       max_radius[pyramid_index(cell_list, cell_index, level)]) / 2

    return distance2 > max_pair_radius^2
end

# Reduce over all neighbors in the finest cells `first_cell:(last_cell - 1)`
@propagate_inbounds function mapreduce_finest_cells(f, op, reduced, first_cell, last_cell,
                                                    neighbor_coords, neighborhood_search,
                                                    point, point_coords, query_radius)
    neighbors = points_in_finest_cells(neighborhood_search.cell_list, first_cell, last_cell)

    return mapreduce_points(f, op, reduced, neighbors, neighbor_coords,
                            neighborhood_search, point, point_coords, query_radius,
                            nothing)
end

function copy_neighborhood_search(nhs::TreeNeighborhoodSearch, search_radius, n_points;
                                  eachpoint = 1:n_points)
    # The search radius is ignored, as the search radii are passed to `initialize_tree!`
    cell_list = copy_cell_list(nhs.cell_list)

    return TreeNeighborhoodSearch{ndims(nhs)}(; cell_list, n_points,
                                              update_strategy = nhs.update_strategy)
end
