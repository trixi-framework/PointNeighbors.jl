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
    TreeNeighborhoodSearch{NDIMS}(; search_radius = 0.0, cell_list, n_points = 0,
                                  pruning = false, neighbor_criterion = MaximumRadius(),
                                  periodic_box = nothing, update_strategy = nothing)

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

Alternatively, the tree can be constructed with [`initialize!`](@ref) and
[`update!`](@ref) like the [`GridNeighborhoodSearch`](@ref).
Then, the neighbor points have no search radius (``h_j = 0``), so that all points ``j`` with
``\| x - y_j \| \leq h`` are neighbors, and the tree has the same structure as with
the uniform search radius `search_radius` for all points.
This is also the default search radius ``h`` of query points.

The points are stored in the leaves of a quadtree (2D) or octree (3D),
see [`TreeCellList`](@ref).
The leaf size is chosen locally based on the search radii of the points,
without any assumptions on the variation of the search radii.
As with the [`GridNeighborhoodSearch`](@ref), a query only considers the ``3^d`` block of
cells around the leaf containing the query point, at the level of this leaf.

To use this neighborhood search on GPUs, move it to the GPU with Adapt.jl
(see the GPU tutorial) and pass GPU arrays to [`initialize_tree!`](@ref)
and [`update_tree!`](@ref).

!!! warning "Experimental implementation"
    This is an experimental feature and may change in any future releases.

# Arguments
- `NDIMS`: Number of dimensions.

# Keywords
- `search_radius = 0.0`: The search radius used by [`initialize!`](@ref) and
                     [`update!`](@ref) (see above), and the default search radius of
                     query points.
                     The default of `0.0` is useful together with
                     [`copy_neighborhood_search`](@ref).
                     It is converted to the element type of the `cell_list`, which is
                     determined by its domain corners.
- `cell_list`:       A [`TreeCellList`](@ref) defining the domain, the size of the root
                     cells and the depth of the tree.
                     With periodicity, the domain must be the periodic box.
- `n_points = 0`:    Maximum total number of points in the neighbor coordinates array.
                     This is used to allocate the lists of points in the leaves.
                     The default of `0` is useful together with
                     [`copy_neighborhood_search`](@ref).
- `pruning = false`:  Skip leaves whose distance to the query point is larger than
                     the pair search radius of the query radius and the maximum search radius
                     of all points in the leaf.
                     With the `CompactVectorOfVectors` backend of the [`TreeCellList`](@ref),
                     leaves that are larger than the query radius are searched
                     hierarchically down to cells of the size of the query radius,
                     skipping all parts of the leaf that are out of reach.
- `neighbor_criterion = MaximumRadius()`: Criterion defining which points are neighbors.
                     See [`MaximumRadius`](@ref), [`AverageRadius`](@ref)
                     and [`QueryRadius`](@ref).
- `periodic_box = nothing`: In order to use a (rectangular) periodic domain, pass a
                     [`PeriodicBox`](@ref). The root cells are adjusted to tile the box
                     exactly, with at least 3 root cells in each dimension.
- `update_strategy = nothing`: Strategy to parallelize the update. The tree is always
                     rebuilt from scratch. Available options are:
    - `nothing`: Automatically choose the best available option.
    - [`ParallelUpdate()`](@ref): Parallel update with the parallelization backend
      passed to [`initialize_tree!`](@ref) and [`update_tree!`](@ref).
      This is the default when available (not with the `Vector{Vector}` backend).
    - [`SerialUpdate()`](@ref)

# Correctness
Denote by ``s_L`` the size of the cells at level ``L``
(the smallest size in all dimensions if the cells are not cubic).
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
# `PRUNING` is a type parameter, so that the pruning code is removed by the compiler
# when pruning is disabled.
struct TreeNeighborhoodSearch{NDIMS, PRUNING, ELTYPE, NC, US, CL, R, PB} <:
       AbstractNeighborhoodSearch
    cell_list          :: CL
    search_radius      :: ELTYPE
    search_radii       :: R  # Copy of the search radii of the neighbor points
    neighbor_criterion :: NC
    periodic_box       :: PB
    update_strategy    :: US
end

function TreeNeighborhoodSearch{NDIMS}(; search_radius = 0.0, cell_list, n_points = 0,
                                       pruning = false,
                                       neighbor_criterion = MaximumRadius(),
                                       periodic_box = nothing,
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

    search_radius_ = convert(eltype(cell_list), search_radius)
    search_radii = zeros(eltype(cell_list), n_points)

    # Adjust the root cells to the periodic box, choose the finest level
    # and allocate the tree data structures with lists for up to `n_points`
    # non-empty leaves
    cell_list_ = build_tree(cell_list, search_radius_, n_points, periodic_box)

    return TreeNeighborhoodSearch{NDIMS, Bool(pruning), typeof(search_radius_),
                                  typeof(neighbor_criterion), typeof(update_strategy),
                                  typeof(cell_list_), typeof(search_radii),
                                  typeof(periodic_box)}(cell_list_, search_radius_,
                                                        search_radii, neighbor_criterion,
                                                        periodic_box, update_strategy)
end

@inline Base.ndims(::TreeNeighborhoodSearch{NDIMS}) where {NDIMS} = NDIMS
@inline uses_pruning(::TreeNeighborhoodSearch{<:Any, PRUNING}) where {PRUNING} = PRUNING
@inline Base.eltype(::TreeNeighborhoodSearch{<:Any, <:Any, ELTYPE}) where {ELTYPE} = ELTYPE

@inline requires_update(::TreeNeighborhoodSearch) = (false, true)

@inline function check_search_radius(neighborhood_search::TreeNeighborhoodSearch,
                                     query_radius)
    # The 3^d block of cells around the query point only contains all neighbors
    # if the query radius does not exceed the size of the root cells
    if query_radius > max_root_radius(neighborhood_search.cell_list)
        error("query `search_radius` must not exceed the root cell size of the tree")
    end

    return nothing
end

function initialize!(neighborhood_search::TreeNeighborhoodSearch,
                     x::AbstractMatrix, y::AbstractMatrix;
                     parallelization_backend = default_backend(x),
                     eachindex_y = axes(y, 2))
    (; search_radii) = neighborhood_search

    # Without search radii of the neighbor points, all neighbor criteria reduce to
    # `distance <= h` for a query point with search radius `h`, like with the
    # `GridNeighborhoodSearch`. By default, `max_level` of the cell list is chosen for the
    # search radius of the neighborhood search, so that the leaves are at the same level
    # as with a search radius of `search_radius` for all points.
    # Only resize when necessary, since resizing GPU arrays always reallocates memory
    if length(search_radii) != size(y, 2)
        resize!(search_radii, size(y, 2))
    end
    fill!(search_radii, 0)

    initialize_tree!(neighborhood_search, y, search_radii;
                     parallelization_backend, eachindex_y)
end

function update!(neighborhood_search::TreeNeighborhoodSearch,
                 x::AbstractMatrix, y::AbstractMatrix;
                 points_moving = (true, true), parallelization_backend = default_backend(x),
                 eachindex_y = axes(y, 2))
    # The coordinates of the first set of points are irrelevant for this NHS.
    # Only update when the second set is moving.
    points_moving[2] || return neighborhood_search

    # The tree is always rebuilt from scratch
    initialize!(neighborhood_search, x, y; parallelization_backend, eachindex_y)
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

With [`ParallelUpdate`](@ref), the keyword argument `parallelization_backend`
can be used to specify a parallelization backend.
See [`@threaded`](@ref) for a list of available backends.

Optionally, when points in `y` are to be ignored, the keyword argument `eachindex_y` can be
passed to specify the indices of the points in `y` that are to be used.

See also [`update_tree!`](@ref).
"""
function initialize_tree!(neighborhood_search::TreeNeighborhoodSearch, y::AbstractMatrix,
                          search_radii::AbstractVector;
                          parallelization_backend = default_backend(y),
                          eachindex_y = axes(y, 2))
    (; cell_list, periodic_box) = neighborhood_search

    if length(search_radii) != size(y, 2)
        throw(DimensionMismatch("the number of search radii must match the number " *
                                "of points in `y`"))
    end
    @boundscheck checkbounds(y, ndims(neighborhood_search), eachindex_y)

    backend = update_backend(neighborhood_search.update_strategy, parallelization_backend)

    # Copy the search radii in parallel.
    # `search_radii` might already be the array of the neighborhood search.
    search_radii_ = neighborhood_search.search_radii
    if search_radii !== search_radii_
        # Only resize when necessary, since resizing GPU arrays always reallocates memory
        if length(search_radii_) != length(search_radii)
            resize!(search_radii_, length(search_radii))
        end
        @threaded backend for i in eachindex(search_radii_, search_radii)
            search_radii_[i] = search_radii[i]
        end
    end

    rebuild!(cell_list, y, search_radii_, eachindex_y, backend, periodic_box)

    return neighborhood_search
end

@inline update_backend(::ParallelUpdate, parallelization_backend) = parallelization_backend
@inline update_backend(::SerialUpdate, _) = SerialBackend()

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
    (; cell_list, search_radii, neighbor_criterion, periodic_box) = neighborhood_search
    (; max_radius_all) = cell_list
    NDIMS = ndims(neighborhood_search)

    query_radius = convert(eltype(neighborhood_search), search_radius)

    # With periodicity, move the query point into the periodic box first
    point_coords_ = periodic_coords(point_coords, periodic_box)
    finest_cell = query_finest_cell(point_coords_, cell_list)
    level = query_level(cell_list, finest_cell, query_radius)

    # The cell at level `level` around which the 3^d block of cells is searched
    cell = query_cell(finest_cell, cell_list, level, periodic_box)

    # Leaves coarser than this level are searched hierarchically with sub-leaf pruning
    subleaf_level = subleaf_pruning_level(neighborhood_search, query_radius)

    # All neighbors are within this radius. Only for points within this radius,
    # we load their search radius to check if they are within the pair search radius.
    max_pair_radius = pair_radius(neighbor_criterion, query_radius, max_radius_all[1])
    skip_neighbor = skip_neighbor_function(neighbor_criterion, query_radius, search_radii)

    # The linear index of a cell is a sum of independent parts for each dimension.
    # `index_parts[dim]` contains the parts of the coordinates `cell - 1`, `cell`
    # and `cell + 1` in dimension `dim`, so that the indices of the 3^d block of cells
    # around the cell are all sums of one part from each dimension.
    index_parts = ntuple(dim -> neighbor_index_parts(cell_list, dim, cell[dim], level,
                                                     periodic_box), Val(NDIMS))

    # The first cell of the 3^d block
    block_min = cell .- 1

    # Note that looping over the positions of the cells in the block (between 0 and 2)
    # requires fewer registers on GPUs than looping over the offsets to `cell`
    # (between -1 and 1).
    reduced = init
    for offset_ in CartesianIndices(ntuple(_ -> 0:2, Val(NDIMS)))
        offset = Tuple(offset_)

        reduced = mapreduce_block_cell(f, op, reduced, block_min .+ offset,
                                       neighbor_index(index_parts, offset), level,
                                       block_min, neighbor_coords, neighborhood_search,
                                       point, point_coords_, query_radius,
                                       max_pair_radius, subleaf_level, skip_neighbor)
    end

    return reduced
end

# The index of the cell `block_min + offset` of the 3^d block, where `offset` is between
# 0 and 2 in each dimension.
# Note that we need the coordinates of the cell in addition to the index, since cells of
# the block can be inside of the same coarser leaf, which must only be visited once
# (see `mapreduce_non_leaf`).
# With periodicity, the cells of the block can be outside of the grid.
# Their index is the index of the corresponding cell inside the grid,
# but their coordinates are outside of the grid, so that they can be used for the geometry.
@inline function neighbor_index(index_parts, offset)
    return sum(ntuple(dim -> index_parts[dim][offset[dim] + 1], Val(length(offset))))
end

# The level of the search block for a point in the finest cell `finest_cell`.
# This is the level of the leaf containing the point, or a coarser level
# if the query radius is larger than the leaf.
@propagate_inbounds function query_level(cell_list, finest_cell, query_radius)
    (; max_level) = cell_list

    # Making the following `@inbounds` is not safe because we don't know if
    # the tree has been initialized correctly.
    level = leaf_level(cell_list, cell_index(cell_list, finest_cell, max_level))

    # Search at the finest level `<= level` with cells that are at least as large
    # as `query_radius`
    while level > 0 && minimum(cell_size(cell_list, level)) < query_radius
        level -= 1
    end

    return level
end

# Return a function that is `true` for neighbors (passed to `mapreduce_points` as
# `skip_neighbor`) that are not within the pair search radius of the query point
@inline function skip_neighbor_function(neighbor_criterion, query_radius, search_radii)
    return @inline function (neighbor, distance2)
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

# With the `MaximumRadius` criterion, all points within the query radius are neighbors,
# so we only have to load the search radii of points outside of the query radius.
# Without search radii of the neighbor points (see `initialize!`), this is never the case.
@inline function skip_neighbor_function(neighbor_criterion::MaximumRadius, query_radius,
                                        search_radii)
    return @inline function (neighbor, distance2)
        distance2 <= query_radius^2 && return false

        return distance2 > search_radii[neighbor]^2
    end
end

# With sub-leaf pruning, leaves coarser than the returned level are searched
# hierarchically down to this level, which is the finest level with cells that are at least
# as large as `query_radius`. This is done when the points in the leaves are sorted by
# finest cell, so that the points of each cell inside a leaf are contiguous.
@inline function subleaf_pruning_level(neighborhood_search, query_radius)
    (; cell_list) = neighborhood_search

    # Never search leaves hierarchically
    uses_subleaf_pruning(neighborhood_search) || return -1

    level = cell_list.max_level
    while level > 0 && minimum(cell_size(cell_list, level)) < query_radius
        level -= 1
    end

    return level
end

@inline function uses_subleaf_pruning(neighborhood_search)
    return uses_pruning(neighborhood_search) && sorts_points(neighborhood_search.cell_list)
end

# Reduce over all neighbors in the cell `cell` at level `level` of the search block,
# which starts at the cell `block_min`.
@propagate_inbounds function mapreduce_block_cell(f, op, reduced, cell, cell_index, level,
                                                  block_min, neighbor_coords,
                                                  neighborhood_search, point, point_coords,
                                                  query_radius, max_pair_radius,
                                                  subleaf_level, skip_neighbor)
    (; cell_list) = neighborhood_search

    first_finest_cell = cell_index << (ndims(cell_list) * (cell_list.max_level - level))
    leaf_level_ = leaf_level(cell_list, first_finest_cell)

    if leaf_level_ == level
        # This cell is a leaf, which is the most common case
        max_pair_radius_ = cell_max_pair_radius(neighborhood_search, cell, cell_index,
                                                level, point_coords, query_radius,
                                                max_pair_radius)
        max_pair_radius_ < 0 && return reduced

        if level < subleaf_level
            return mapreduce_subcells(f, op, reduced, first_finest_cell, cell, level,
                                      subleaf_level, neighbor_coords, neighborhood_search,
                                      point, point_coords, query_radius, max_pair_radius,
                                      skip_neighbor)
        end

        return mapreduce_leaf(f, op, reduced, first_finest_cell, neighbor_coords,
                              neighborhood_search, point, point_coords, max_pair_radius_,
                              skip_neighbor)
    end

    return mapreduce_non_leaf(f, op, reduced, cell, cell_index, level, leaf_level_,
                              first_finest_cell, block_min, neighbor_coords,
                              neighborhood_search, point, point_coords, query_radius,
                              max_pair_radius, subleaf_level, skip_neighbor)
end

# Reduce over all neighbors in a cell of the search block that is not a leaf.
# This cell can be inside a coarser leaf or be split into finer leaves.
@propagate_inbounds function mapreduce_non_leaf(f, op, reduced, cell, cell_index, level,
                                                leaf_level_, first_finest_cell, block_min,
                                                neighbor_coords, neighborhood_search,
                                                point, point_coords, query_radius,
                                                max_pair_radius, subleaf_level,
                                                skip_neighbor)
    (; cell_list, periodic_box) = neighborhood_search

    # With periodicity, `cell` can be outside of the grid and `cell_` is the corresponding
    # cell inside the grid
    cell_ = periodic_cell(cell, n_cells_per_dimension(cell_list, level), periodic_box)

    if leaf_level_ < level
        # This cell is inside a coarser leaf, which might contain multiple cells of the
        # search block. Only visit this leaf once, from the first cell of the block
        # inside this leaf (with the smallest coordinates in each dimension).
        level_difference = level - leaf_level_
        leaf_min = cell .- (cell_ .- ((cell_ .>> level_difference) .<< level_difference))
        if cell != max.(leaf_min, block_min)
            return reduced
        end

        leaf_cell = leaf_min .>> level_difference
        max_pair_radius_ = cell_max_pair_radius(neighborhood_search, leaf_cell,
                                                cell_index >>
                                                (ndims(cell_list) * level_difference),
                                                leaf_level_, point_coords, query_radius,
                                                max_pair_radius)
        max_pair_radius_ < 0 && return reduced

        if leaf_level_ < subleaf_level
            return mapreduce_subcells(f, op, reduced,
                                      leaf_first_cell(cell_list, first_finest_cell),
                                      leaf_cell, leaf_level_, subleaf_level,
                                      neighbor_coords, neighborhood_search, point,
                                      point_coords, query_radius, max_pair_radius,
                                      skip_neighbor)
        end

        return mapreduce_leaf(f, op, reduced,
                              leaf_first_cell(cell_list, first_finest_cell),
                              neighbor_coords, neighborhood_search, point, point_coords,
                              max_pair_radius_, skip_neighbor)
    end

    # This cell is split. Skip it if all leaves inside are out of reach.
    cell_max_pair_radius(neighborhood_search, cell, cell_index, level, point_coords,
                         query_radius, max_pair_radius) < 0 && return reduced

    if uses_subleaf_pruning(neighborhood_search)
        # The points of each cell are contiguous, so we can search the cells inside this
        # cell hierarchically, independent of the leaves
        return mapreduce_subcells(f, op, reduced, first_finest_cell, cell, level,
                                  subleaf_level, neighbor_coords, neighborhood_search,
                                  point, point_coords, query_radius, max_pair_radius,
                                  skip_neighbor)
    end

    # Loop over all leaves inside this cell.
    # The finest cells of each leaf are contiguous, so we can jump from leaf to leaf.
    finest_cell = first_finest_cell
    end_cell = first_finest_cell + n_finest_cells(cell_list, level)
    while finest_cell < end_cell
        leaf_level_ = leaf_level(cell_list, finest_cell)

        if uses_pruning(neighborhood_search)
            leaf_index = finest_cell >>
                         (ndims(cell_list) * (cell_list.max_level - leaf_level_))
            # With periodicity, shift the leaf to the periodic image next to `cell`
            leaf_cell = cell_coords_from_index(cell_list, leaf_index, leaf_level_) .+
                        ((cell .- cell_) .<< (leaf_level_ - level))
            max_pair_radius_ = cell_max_pair_radius(neighborhood_search, leaf_cell,
                                                    leaf_index, leaf_level_,
                                                    point_coords, query_radius,
                                                    max_pair_radius)

            if max_pair_radius_ >= 0 && leaf_level_ < subleaf_level
                reduced = mapreduce_subcells(f, op, reduced, finest_cell, leaf_cell,
                                             leaf_level_, subleaf_level, neighbor_coords,
                                             neighborhood_search, point, point_coords,
                                             query_radius, max_pair_radius,
                                             skip_neighbor)
                # Skip this leaf below
                max_pair_radius_ = -one(max_pair_radius_)
            end
        else
            max_pair_radius_ = max_pair_radius
        end

        if max_pair_radius_ >= 0
            reduced = mapreduce_leaf(f, op, reduced, finest_cell, neighbor_coords,
                                     neighborhood_search, point, point_coords,
                                     max_pair_radius_, skip_neighbor)
        end

        finest_cell += n_finest_cells(cell_list, leaf_level_)
    end

    return reduced
end

# With pruning, return the maximum search radius of a pair of the query point and any
# point in the cell `cell` with index `cell_index` at level `level`, or a negative value
# if all points in this cell are too far away from the query point to be neighbors.
# Without pruning, just return `max_pair_radius`, the maximum pair radius of all points.
@propagate_inbounds function cell_max_pair_radius(neighborhood_search, cell, cell_index,
                                                  level, point_coords, query_radius,
                                                  max_pair_radius)
    uses_pruning(neighborhood_search) || return max_pair_radius

    (; cell_list, neighbor_criterion) = neighborhood_search
    (; max_radius) = cell_list

    # All pair search radii are monotonically increasing in the radius of the neighbor
    max_pair_radius_ = pair_radius(neighbor_criterion, query_radius,
                                   max_radius[pyramid_index(cell_list, cell_index, level)])

    if is_out_of_reach(cell_list, cell, level, point_coords, max_pair_radius_)
        return -one(max_pair_radius_)
    end

    return max_pair_radius_
end

# Check if all points in the cell `cell` (zero-based Cartesian coordinates)
# at level `level` are too far away from the query point to be neighbors.
# This is the case when the minimum distance from the query point to the cell is larger
# than the maximum pair search radius in this cell.
# With periodicity, `cell` can be outside of the grid. Then, this is the periodic image
# of the cell next to the query point inside the periodic box.
@inline function is_out_of_reach(cell_list, cell, level, point_coords, max_pair_radius)
    (; grid_min_corner, actual_root_cell_size, n_roots) = cell_list
    ELTYPE = eltype(cell_list)

    # Enlarge the cell by a safety margin to account for rounding errors when assigning
    # points to cells. These are in the order of machine precision times the domain size.
    margin = 16 * eps(ELTYPE) * maximum(actual_root_cell_size .* n_roots)

    size_ = cell_size(cell_list, level)
    cell_min = grid_min_corner .+ SVector(cell) .* size_ .- margin
    cell_max = cell_min .+ size_ .+ 2 * margin

    # Distance vector from the query point to the closest point in the cell
    distance_vector = max.(cell_min .- point_coords, point_coords .- cell_max, 0)
    distance2 = dot(distance_vector, distance_vector)

    return distance2 > max_pair_radius^2
end

# Reduce over all neighbors in the leaf with the first finest cell `first_finest_cell`
@propagate_inbounds function mapreduce_leaf(f, op, reduced, first_finest_cell,
                                            neighbor_coords, neighborhood_search, point,
                                            point_coords, max_pair_radius, skip_neighbor)
    neighbors = points_in_leaf(neighborhood_search.cell_list, first_finest_cell)

    return mapreduce_points(f, op, reduced, neighbors, neighbor_coords,
                            neighborhood_search, point, point_coords, max_pair_radius,
                            skip_neighbor)
end

# Reduce over all neighbors in the list `neighbors` within the radius `max_pair_radius`
# that are not skipped by `skip_neighbor(neighbor, distance2)`
@propagate_inbounds function mapreduce_points(f, op, reduced, neighbors, neighbor_coords,
                                              neighborhood_search, point, point_coords,
                                              max_pair_radius, skip_neighbor)
    (; periodic_box) = neighborhood_search

    for neighbor_ in eachindex(neighbors)
        neighbor = @inbounds neighbors[neighbor_]

        # Making the following `@inbounds` is not safe because we don't know
        # if `neighbor` (extracted from the cell list) is in bounds.
        neighbor_point_coords = extract_svector(neighbor_coords,
                                                Val(ndims(neighborhood_search)), neighbor)

        pos_diff = convert.(eltype(neighborhood_search),
                            point_coords - neighbor_point_coords)
        distance2 = dot(pos_diff, pos_diff)

        (pos_diff,
         distance2) = compute_periodic_distance(pos_diff, distance2,
                                                max_pair_radius, periodic_box)

        if distance2 <= max_pair_radius^2
            @inline(skip_neighbor(neighbor, distance2)) && continue

            distance = sqrt(distance2)

            # Inline to avoid loss of performance compared to not using this function
            # and unrolling everything.
            value = @inline f(point, neighbor, pos_diff, distance)
            reduced = @inline op(reduced, value)
        end
    end

    return reduced
end

# Reduce over all neighbors in the cell (usually a leaf) at level `leaf_level` with
# the first finest cell `first_finest_cell` and the (with periodicity possibly shifted)
# cell coordinates `leaf_cell` by traversing the cells inside down to level `subleaf_level`.
# Cells that are out of reach are skipped together with all their children.
# This requires the points to be sorted by finest cell, see `points_in_finest_cells`.
@propagate_inbounds function mapreduce_subcells(f, op, reduced, first_finest_cell,
                                                leaf_cell, leaf_level, subleaf_level,
                                                neighbor_coords, neighborhood_search,
                                                point, point_coords, query_radius,
                                                max_pair_radius, skip_neighbor)
    (; cell_list) = neighborhood_search
    (; max_level, n_points) = cell_list
    NDIMS = ndims(cell_list)

    # Depth-first traversal of the cells inside the leaf without a stack.
    # In Morton order, each cell is a contiguous range of finest cells, so we only have to
    # keep track of the next finest cell to be visited.
    end_cell = first_finest_cell + n_finest_cells(cell_list, leaf_level)
    finest_cell = first_finest_cell
    while finest_cell < end_cell
        # Start at the coarsest level below the leaf with a cell starting at `finest_cell`
        local_index = finest_cell - first_finest_cell
        level = max(leaf_level + 1, max_level - div(trailing_zeros(local_index), NDIMS))

        while true
            shift = NDIMS * (max_level - level)
            cell_index_ = finest_cell >> shift

            # Coordinates of this cell, computed from its index inside the leaf
            cell = (leaf_cell .<< (level - leaf_level)) .+
                   morton_coords(local_index >> shift, Val(NDIMS))

            # Skip empty cells and cells out of reach
            n_points[pyramid_index(cell_list, cell_index_, level)] == 0 && break
            max_pair_radius_ = cell_max_pair_radius(neighborhood_search, cell, cell_index_,
                                                    level, point_coords, query_radius,
                                                    max_pair_radius)
            max_pair_radius_ < 0 && break

            if level >= subleaf_level
                neighbors = points_in_finest_cells(cell_list, finest_cell, 1 << shift)
                reduced = mapreduce_points(f, op, reduced, neighbors, neighbor_coords,
                                           neighborhood_search, point, point_coords,
                                           max_pair_radius_, skip_neighbor)
                break
            end

            # Continue with the first child
            level += 1
        end

        # Continue after the last visited cell
        finest_cell += n_finest_cells(cell_list, level)
    end

    return reduced
end

function copy_neighborhood_search(nhs::TreeNeighborhoodSearch, search_radius, n_points;
                                  eachpoint = 1:n_points)
    # Use root cells of the size of the search radius.
    cell_list = copy_cell_list(nhs.cell_list, search_radius)

    return TreeNeighborhoodSearch{ndims(nhs)}(; search_radius, cell_list, n_points,
                                              pruning = uses_pruning(nhs),
                                              nhs.neighbor_criterion,
                                              periodic_box = nhs.periodic_box,
                                              update_strategy = nhs.update_strategy)
end
