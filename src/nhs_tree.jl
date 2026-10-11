abstract type AbstractNeighborCriterion end

@doc raw"""
    MaximumRadius()

Neighbor criterion for the [`TreeNeighborhoodSearch`](@ref).
A point ``j`` with search radius ``h_j`` is a neighbor of a query point ``x`` with search
radius ``h`` if
```math
\| x - y_j \| \leq \max(h, h_j).
```
This is the safe default, which finds all pairs of points within the search radius of
either point. The other criteria find a subset of these pairs.

See also [`AverageRadius`](@ref) and [`QueryRadius`](@ref).
"""
struct MaximumRadius <: AbstractNeighborCriterion end

@doc raw"""
    AverageRadius()

Neighbor criterion for the [`TreeNeighborhoodSearch`](@ref).
A point ``j`` with search radius ``h_j`` is a neighbor of a query point ``x`` with search
radius ``h`` if
```math
\| x - y_j \| \leq \frac{h + h_j}{2}.
```

See also [`MaximumRadius`](@ref) and [`QueryRadius`](@ref).
"""
struct AverageRadius <: AbstractNeighborCriterion end

@doc raw"""
    QueryRadius()

Neighbor criterion for the [`TreeNeighborhoodSearch`](@ref).
A point ``j`` is a neighbor of a query point ``x`` with search radius ``h`` if
```math
\| x - y_j \| \leq h,
```
independent of the search radius ``h_j`` of the point ``j``.
Note that the search radii of the neighbor points are still used to build the tree.

See also [`MaximumRadius`](@ref) and [`AverageRadius`](@ref).
"""
struct QueryRadius <: AbstractNeighborCriterion end

# The search radius for a pair of a query point with search radius `h` and a point
# with search radius `h_j`
@inline pair_radius(::MaximumRadius, h, h_j) = max(h, h_j)
@inline pair_radius(::AverageRadius, h, h_j) = (h + h_j) / 2
@inline pair_radius(::QueryRadius, h, h_j) = h

@doc raw"""
    TreeNeighborhoodSearch{NDIMS}(; cell_list, n_points = 0,
                                  neighbor_criterion = MaximumRadius(),
                                  update_strategy = nothing)

Tree-based neighborhood search with a variable search radius ``h_j`` for each point ``j``.
For a query point ``x`` with search radius ``h``, all points ``j`` with
```math
\| x - y_j \| \leq r(h, h_j)
```
are considered neighbors, where the pair search radius ``r(h, h_j)`` is defined by the
`neighbor_criterion`:
- [`MaximumRadius()`](@ref): ``r(h, h_j) = \max(h, h_j)``,
- [`AverageRadius()`](@ref): ``r(h, h_j) = (h + h_j) / 2``,
- [`QueryRadius()`](@ref): ``r(h, h_j) = h``.

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

!!! warning "Experimental implementation"
    This is an experimental feature and may change in any future releases.

# Arguments
- `NDIMS`: Number of dimensions.

# Keywords
- `cell_list`:       A [`TreeCellList`](@ref) defining the domain and the depth of the tree.
- `n_points = 0`:    Total number of points in the neighbor coordinates array.
                     The default of `0` is useful together with
                     [`copy_neighborhood_search`](@ref).
- `neighbor_criterion = MaximumRadius()`: Criterion defining which points are neighbors.
                     See [`MaximumRadius`](@ref), [`AverageRadius`](@ref)
                     and [`QueryRadius`](@ref).
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
we have ``\| x - y_j \| > s_{m+1} \geq \max(h, h_j) \geq r(h, h_j)``.
Therefore, ``j`` is not a neighbor of ``x``.

For a point ``x = y_i`` of the tree with ``h = h_i``, the leaf containing ``x`` satisfies
``s_{L_x} \geq h``, so we always search at the level of the leaf, like with the
[`GridNeighborhoodSearch`](@ref).
"""
struct TreeNeighborhoodSearch{NDIMS, NC, US, CL, R} <: AbstractNeighborhoodSearch
    cell_list          :: CL
    search_radii       :: R  # Copy of the search radii of the neighbor points
    neighbor_criterion :: NC
    update_strategy    :: US
end

function TreeNeighborhoodSearch{NDIMS}(; cell_list, n_points = 0,
                                       neighbor_criterion = MaximumRadius(),
                                       update_strategy = nothing) where {NDIMS}
    if !(cell_list isa TreeCellList)
        throw(ArgumentError("`TreeNeighborhoodSearch` requires a `TreeCellList`"))
    end

    if ndims(cell_list) != NDIMS
        throw(ArgumentError("a $(NDIMS)D cell list is required for " *
                            "a TreeNeighborhoodSearch{$(NDIMS)}"))
    end

    if !(neighbor_criterion isa AbstractNeighborCriterion)
        throw(ArgumentError("`neighbor_criterion` must be `MaximumRadius()`, " *
                            "`AverageRadius()` or `QueryRadius()`"))
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

    return TreeNeighborhoodSearch{NDIMS, typeof(neighbor_criterion),
                                  typeof(update_strategy), typeof(cell_list),
                                  typeof(search_radii)}(cell_list, search_radii,
                                                        neighbor_criterion,
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

# Note that calling this function with `@inbounds` is not safe.
# See the comments in `foreach_neighbor_unsafe`.
@propagate_inbounds function mapreduce_neighbor_inner(f, op, neighbor_coords,
                                                      neighborhood_search::TreeNeighborhoodSearch,
                                                      point, point_coords,
                                                      search_radius, init)
    (; cell_list, search_radii, neighbor_criterion) = neighborhood_search
    (; max_level, max_radius_all) = cell_list
    NDIMS = ndims(neighborhood_search)

    query_radius = convert(eltype(neighborhood_search), search_radius)
    finest_cell = cell_coords(point_coords, cell_list, max_level)
    level = query_level(cell_list, finest_cell, query_radius)
    block_min,
    block_max = search_block(cell_list, finest_cell, level, point_coords,
                             query_radius, neighbor_criterion)

    # All neighbors are within this radius. Only for points within this radius,
    # we load their search radius to check if they are within the pair search radius.
    max_pair_radius = pair_radius(neighbor_criterion, query_radius, max_radius_all[1])
    skip_neighbor = skip_neighbor_function(neighbor_criterion, query_radius, search_radii)

    reduced = init
    if all(block_max .- block_min .< 3)
        # Usually, the block has at most 3 cells in each dimension. The linear cell index
        # is a sum of independent parts for each dimension, so we only compute these
        # parts once for each coordinate in each dimension.
        parts = ntuple(dim -> ntuple(i -> cell_index_part(cell_list, dim,
                                                          block_min[dim] + i - 1, level),
                                     Val(3)), Val(NDIMS))

        for offset_ in CartesianIndices(ntuple(dim -> 0:(block_max[dim] - block_min[dim]),
                                    Val(NDIMS)))
            offset = Tuple(offset_)
            cell = block_min .+ offset
            cell_index_ = sum(ntuple(dim -> parts[dim][offset[dim] + 1], Val(NDIMS)))

            reduced = mapreduce_block_cell(f, op, reduced, cell, cell_index_, level,
                                           block_min, neighbor_coords, neighborhood_search,
                                           point, point_coords, max_pair_radius,
                                           skip_neighbor)
        end
    else
        # Larger blocks at level 0 for large query radii
        for cell_ in CartesianIndices(ntuple(dim -> block_min[dim]:block_max[dim],
                                    Val(NDIMS)))
            cell = Tuple(cell_)
            cell_index_ = cell_index(cell_list, cell, level)

            reduced = mapreduce_block_cell(f, op, reduced, cell, cell_index_, level,
                                           block_min, neighbor_coords, neighborhood_search,
                                           point, point_coords, max_pair_radius,
                                           skip_neighbor)
        end
    end

    return reduced
end

# The level of the search block for a point in the finest cell `finest_cell`.
# This is the level of the leaf containing the point, or a coarser level
# if the query radius is larger than the leaf.
@propagate_inbounds function query_level(cell_list, finest_cell, query_radius)
    (; max_level) = cell_list

    # Points outside the grid are treated as if they were in a leaf at level 0
    is_in_grid(finest_cell, cell_list, max_level) || return 0

    # Making the following `@inbounds` is not safe because we don't know if
    # the tree has been initialized correctly.
    level = leaf_level(cell_list, cell_index(cell_list, finest_cell, max_level))

    # Search at the finest level `<= level` with cells that are at least as large
    # as `query_radius`
    while level > 0 && cell_size(cell_list, level) < query_radius
        level -= 1
    end

    return level
end

# Return a function that is `true` for neighbors (passed to `mapreduce_points` as
# `skip_neighbor`) that are not within the pair search radius of the query point
@inline function skip_neighbor_function(neighbor_criterion, query_radius, search_radii)
    return @inline function (neighbor, _, distance2)
        pair_radius_ = pair_radius(neighbor_criterion, query_radius,
                                   search_radii[neighbor])
        return distance2 > pair_radius_^2
    end
end

# With the `QueryRadius` criterion, all points within the query radius are neighbors,
# so we don't need to load the search radii of the neighbors
@inline function skip_neighbor_function(::QueryRadius, query_radius, search_radii)
    return Returns(false)
end

# The first and the last cell (zero-based Cartesian coordinates) of the block of cells
# at level `level` that has to be searched for neighbors of a point.
@propagate_inbounds function search_block(cell_list, finest_cell, level, point_coords,
                                          query_radius, neighbor_criterion)
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
    max_pair_radius = pair_radius(neighbor_criterion, query_radius, max_radius_all[1])
    block_min = Tuple(floor_to_int.((point_coords .- max_pair_radius .- min_corner) ./
                                    root_cell_size))
    block_max = Tuple(floor_to_int.((point_coords .+ max_pair_radius .- min_corner) ./
                                    root_cell_size))

    return max.(block_min, 0), min.(block_max, n_roots .- 1)
end

# Reduce over all neighbors in the cell `cell` at level `level` of the search block,
# which starts at the cell `block_min`.
@propagate_inbounds function mapreduce_block_cell(f, op, reduced, cell, cell_index, level,
                                                  block_min, neighbor_coords,
                                                  neighborhood_search, point, point_coords,
                                                  max_pair_radius, skip_neighbor)
    (; cell_list) = neighborhood_search

    first_finest_cell = cell_index << (ndims(cell_list) * (cell_list.max_level - level))
    leaf_level_ = leaf_level(cell_list, first_finest_cell)

    if leaf_level_ == level
        # This cell is a leaf, which is the most common case
        return mapreduce_leaf(f, op, reduced, first_finest_cell, neighbor_coords,
                              neighborhood_search, point, point_coords, max_pair_radius,
                              skip_neighbor)
    end

    return mapreduce_non_leaf(f, op, reduced, cell, level, leaf_level_, first_finest_cell,
                              block_min, neighbor_coords, neighborhood_search, point,
                              point_coords, max_pair_radius, skip_neighbor)
end

# Reduce over all neighbors in a cell of the search block that is not a leaf.
# This cell can be inside a coarser leaf or be split into finer leaves.
@propagate_inbounds function mapreduce_non_leaf(f, op, reduced, cell, level, leaf_level_,
                                                first_finest_cell, block_min,
                                                neighbor_coords, neighborhood_search,
                                                point, point_coords, max_pair_radius,
                                                skip_neighbor)
    (; cell_list) = neighborhood_search

    if leaf_level_ < level
        # This cell is inside a coarser leaf, which might contain multiple cells of the
        # search block. Only visit this leaf once, from the first cell of the block
        # inside this leaf (with the smallest coordinates in each dimension).
        level_difference = level - leaf_level_
        leaf_min = (cell .>> level_difference) .<< level_difference
        if cell != max.(leaf_min, block_min)
            return reduced
        end

        return mapreduce_leaf(f, op, reduced,
                              leaf_first_cell(cell_list, first_finest_cell),
                              neighbor_coords, neighborhood_search, point, point_coords,
                              max_pair_radius, skip_neighbor)
    end

    # This cell is split. Loop over all leaves inside this cell.
    # The finest cells of each leaf are contiguous, so we can jump from leaf to leaf.
    finest_cell = first_finest_cell
    end_cell = first_finest_cell + n_finest_cells(cell_list, level)
    while finest_cell < end_cell
        reduced = mapreduce_leaf(f, op, reduced, finest_cell, neighbor_coords,
                                 neighborhood_search, point, point_coords, max_pair_radius,
                                 skip_neighbor)

        finest_cell += n_finest_cells(cell_list, leaf_level(cell_list, finest_cell))
    end

    return reduced
end

# Reduce over all neighbors in the leaf with the first finest cell `first_finest_cell`
@propagate_inbounds function mapreduce_leaf(f, op, reduced, first_finest_cell,
                                            neighbor_coords, neighborhood_search, point,
                                            point_coords, max_pair_radius, skip_neighbor)
    neighbors = points_in_leaf(neighborhood_search.cell_list, first_finest_cell)

    return mapreduce_points(f, op, reduced, neighbors, neighbor_coords,
                            neighborhood_search, point, point_coords, max_pair_radius,
                            nothing, skip_neighbor)
end

function copy_neighborhood_search(nhs::TreeNeighborhoodSearch, search_radius, n_points;
                                  eachpoint = 1:n_points)
    # The search radius is ignored, as the search radii are passed to `initialize_tree!`
    cell_list = copy_cell_list(nhs.cell_list)

    return TreeNeighborhoodSearch{ndims(nhs)}(; cell_list, n_points,
                                              nhs.neighbor_criterion,
                                              update_strategy = nhs.update_strategy)
end
