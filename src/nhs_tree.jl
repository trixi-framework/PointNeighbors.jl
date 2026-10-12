@doc raw"""
    TreeNeighborhoodSearch{NDIMS}(; search_radius = 0.0, cell_list,
                                  update_strategy = nothing)

Tree-based neighborhood search with a uniform search radius.
The points are stored in the leaves of a quadtree (2D) or octree (3D),
see [`TreeCellList`](@ref).

With a uniform search radius, all leaves are at the same level, which is the finest level
with cells that are at least as large as the search radius.
As with the [`GridNeighborhoodSearch`](@ref), a query only considers the ``3^d`` block of
leaves around the leaf containing the query point.
The only difference to the [`GridNeighborhoodSearch`](@ref) with
[`FullGridCellList`](@ref) is the order of the cells, which are sorted in Morton order
inside each root cell of the tree.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in any future releases.

# Arguments
- `NDIMS`: Number of dimensions.

# Keywords
- `search_radius = 0.0`: The fixed search radius. The default of `0.0` is useful together
                         with [`copy_neighborhood_search`](@ref).
                         The search radius must not exceed the root cell size of the
                         `cell_list`.
- `cell_list`:           A [`TreeCellList`](@ref) defining the domain and the depth
                         of the tree.
- `update_strategy = nothing`: Strategy to parallelize the update. The tree is always
                         rebuilt from scratch. Available options are:
    - `nothing`: Automatically choose the best available option.
    - [`SerialUpdate()`](@ref)
"""
struct TreeNeighborhoodSearch{NDIMS, ELTYPE, CL, US} <: AbstractNeighborhoodSearch
    cell_list       :: CL
    search_radius   :: ELTYPE
    update_strategy :: US
end

function TreeNeighborhoodSearch{NDIMS}(; search_radius = 0.0, cell_list,
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

    search_radius_ = convert(eltype(cell_list), search_radius)

    # Choose the level of the leaves and allocate a list for each leaf
    cell_list_ = allocate_leaves(cell_list, search_radius_)

    return TreeNeighborhoodSearch{NDIMS, typeof(search_radius_), typeof(cell_list_),
                                  typeof(update_strategy)}(cell_list_, search_radius_,
                                                           update_strategy)
end

@inline Base.ndims(::TreeNeighborhoodSearch{NDIMS}) where {NDIMS} = NDIMS
@inline Base.eltype(::TreeNeighborhoodSearch{<:Any, ELTYPE}) where {ELTYPE} = ELTYPE

@inline requires_update(::TreeNeighborhoodSearch) = (false, true)

function initialize!(neighborhood_search::TreeNeighborhoodSearch,
                     x::AbstractMatrix, y::AbstractMatrix;
                     parallelization_backend = default_backend(x),
                     eachindex_y = axes(y, 2))
    @boundscheck checkbounds(y, ndims(neighborhood_search), eachindex_y)

    rebuild!(neighborhood_search.cell_list, y, eachindex_y)

    return neighborhood_search
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

# Note that calling this function with `@inbounds` is not safe.
# See the comments in `foreach_neighbor_unsafe`.
@propagate_inbounds function mapreduce_neighbor_inner(f, op, neighbor_coords,
                                                      neighborhood_search::TreeNeighborhoodSearch,
                                                      point, point_coords,
                                                      search_radius, init)
    (; cell_list) = neighborhood_search
    (; leaf_level) = cell_list
    NDIMS = ndims(neighborhood_search)

    # Zero-based Cartesian coordinates of the leaf containing the point
    leaf = cell_coords(point_coords, cell_list, leaf_level)

    # The 3^d block of leaves around this leaf, clipped to the grid of leaves.
    # For points far outside the grid, this block is empty.
    # Unlike the `FullGridCellList`, we don't use a padding layer of empty cells around
    # the grid, which would require additional root cells.
    # The clipping is not measurably slower than a padding.
    n_leaves = n_cells_per_dimension(cell_list, leaf_level)
    block_min = max.(leaf .- 1, 0)
    block_max = min.(leaf .+ 1, n_leaves .- 1)

    # Computing the index of a leaf with `cell_index` is more expensive than computing
    # a linear index in a grid, which makes the query up to 25% slower when we compute it
    # for each of the 3^d leaves in the block.
    # Instead, we use that the index is a sum of one part for each dimension,
    # and each part only depends on the coordinate in this dimension.
    # For example, in 2D inside a single root cell, the bits of the x-coordinate are
    # at the even bit positions of the index and the bits of the y-coordinate at the odd
    # ones. For the cell `(x, y) = (0b11, 0b01)`, the x-part is `0b0101`, the y-part is
    # `0b0010`, and the index is their sum `0b0111`.
    # We compute the parts for the 3 coordinates of the block in each dimension only once.
    # `index_parts[dim][i]` is the part for the `i`-th coordinate of the block
    # in dimension `dim`. If the block is clipped, the last parts are not used.
    index_parts = ntuple(dim -> ntuple(i -> cell_index_part(cell_list, dim,
                                                            block_min[dim] + i - 1,
                                                            leaf_level),
                                       Val(3)), Val(NDIMS))

    block_size = block_max .- block_min .+ 1

    reduced = init
    for position_ in CartesianIndices(ntuple(dim -> 1:block_size[dim], Val(NDIMS)))
        # Position of the leaf in the block, between 1 and 3 in each dimension
        position = Tuple(position_)
        leaf_index = sum(ntuple(dim -> index_parts[dim][position[dim]], Val(NDIMS)))

        # Making the following `@inbounds` is not safe because we don't know if
        # the cell list has been initialized correctly.
        neighbors = points_in_leaf(cell_list, leaf_index)

        reduced = mapreduce_leaf(f, op, reduced, neighbors, neighbor_coords,
                                 neighborhood_search, point, point_coords, search_radius)
    end

    return reduced
end

# Reduce over all neighbors in the list `neighbors`
@propagate_inbounds function mapreduce_leaf(f, op, reduced, neighbors, neighbor_coords,
                                            neighborhood_search, point, point_coords,
                                            search_radius)
    for neighbor_ in eachindex(neighbors)
        neighbor = @inbounds neighbors[neighbor_]

        # Making the following `@inbounds` is not safe because we don't know
        # if `neighbor` (extracted from the cell list) is in bounds.
        neighbor_point_coords = extract_svector(neighbor_coords,
                                                Val(ndims(neighborhood_search)), neighbor)

        pos_diff = convert.(eltype(neighborhood_search),
                            point_coords - neighbor_point_coords)
        distance2 = dot(pos_diff, pos_diff)

        if distance2 <= search_radius^2
            distance = sqrt(distance2)

            # Inline to avoid loss of performance compared to not using this function
            # and unrolling everything.
            value = @inline f(point, neighbor, pos_diff, distance)
            reduced = @inline op(reduced, value)
        end
    end

    return reduced
end

function copy_neighborhood_search(nhs::TreeNeighborhoodSearch, search_radius, n_points;
                                  eachpoint = 1:n_points)
    cell_list = copy_cell_list(nhs.cell_list)

    return TreeNeighborhoodSearch{ndims(nhs)}(; search_radius, cell_list,
                                              update_strategy = nhs.update_strategy)
end
