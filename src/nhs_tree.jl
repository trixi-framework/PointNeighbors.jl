@doc raw"""
    TreeNeighborhoodSearch{NDIMS}(; search_radius = 0.0, cell_list,
                                  periodic_box = nothing, update_strategy = nothing)

Tree-based neighborhood search with a uniform search radius.
The points are stored in the leaves of a quadtree (2D) or octree (3D),
see [`TreeCellList`](@ref).

The only difference to the [`GridNeighborhoodSearch`](@ref) with
[`FullGridCellList`](@ref) is the order of the cells, which are sorted in Morton order
inside each root cell of the tree.
This neighborhood search will later support variable search radius.
Until then, prefer the [`GridNeighborhoodSearch`](@ref) for better performance.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in any future releases.

# Arguments
- `NDIMS`: Number of dimensions.

# Keywords
- `search_radius = 0.0`: The fixed search radius. The default of `0.0` is useful together
                         with [`copy_neighborhood_search`](@ref).
                         The search radius must not exceed the root cell size of the
                         `cell_list`.
- `cell_list`:           A [`TreeCellList`](@ref) defining the domain and the size of the
                         root cells. With periodicity, the domain must be the periodic box.
- `periodic_box = nothing`: In order to use a (rectangular) periodic domain, pass a
                         [`PeriodicBox`](@ref).
- `update_strategy = nothing`: Strategy to parallelize the update. The tree is always
                         rebuilt from scratch. Available options are:
    - `nothing`: Automatically choose the best available option.
    - [`SerialUpdate()`](@ref)
"""
struct TreeNeighborhoodSearch{NDIMS, ELTYPE, CL, PB, US} <: AbstractNeighborhoodSearch
    cell_list       :: CL
    search_radius   :: ELTYPE
    periodic_box    :: PB
    update_strategy :: US
end

function TreeNeighborhoodSearch{NDIMS}(; search_radius = 0.0, cell_list,
                                       periodic_box = nothing,
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
    cell_list_ = allocate_leaves(cell_list, search_radius_, periodic_box)

    return TreeNeighborhoodSearch{NDIMS, typeof(search_radius_), typeof(cell_list_),
                                  typeof(periodic_box),
                                  typeof(update_strategy)}(cell_list_, search_radius_,
                                                           periodic_box, update_strategy)
end

@inline Base.ndims(::TreeNeighborhoodSearch{NDIMS}) where {NDIMS} = NDIMS
@inline Base.eltype(::TreeNeighborhoodSearch{<:Any, ELTYPE}) where {ELTYPE} = ELTYPE

@inline requires_update(::TreeNeighborhoodSearch) = (false, true)

function initialize!(neighborhood_search::TreeNeighborhoodSearch,
                     x::AbstractMatrix, y::AbstractMatrix;
                     parallelization_backend = default_backend(x),
                     eachindex_y = axes(y, 2))
    (; cell_list, periodic_box) = neighborhood_search

    @boundscheck checkbounds(y, ndims(neighborhood_search), eachindex_y)

    rebuild!(cell_list, y, eachindex_y, periodic_box)

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
    (; cell_list, periodic_box) = neighborhood_search
    NDIMS = ndims(neighborhood_search)

    # Zero-based Cartesian coordinates of the leaf containing the point.
    # With periodicity, move the point into the periodic box first.
    leaf = leaf_coords(periodic_coords(point_coords, periodic_box), cell_list,
                       periodic_box)

    # The linear index of a leaf is a sum of independent parts for each dimension.
    # `index_parts[dim]` contains the parts of the coordinates `leaf - 1`, `leaf`
    # and `leaf + 1` in dimension `dim`, so that the indices of the 3^d block of leaves
    # around the leaf are all sums of one part from each dimension.
    index_parts = ntuple(dim -> neighbor_index_parts(cell_list, dim, leaf[dim],
                                                     periodic_box), Val(NDIMS))

    reduced = init
    for offset in CartesianIndices(ntuple(_ -> -1:1, Val(NDIMS)))
        leaf_index = neighbor_index(index_parts, Tuple(offset))

        # Making the following `@inbounds` is not safe because we don't know if
        # the cell list has been initialized correctly.
        neighbors = points_in_leaf(cell_list, leaf_index)

        reduced = mapreduce_leaf(f, op, reduced, neighbors, neighbor_coords,
                                 neighborhood_search, point, point_coords, search_radius)
    end

    return reduced
end

# The index of the leaf `leaf + offset`, where `offset` is between -1 and 1
# in each dimension.
@inline function neighbor_index(index_parts, offset)
    return sum(ntuple(dim -> index_parts[dim][offset[dim] + 2], Val(length(offset))))
end

# Reduce over all neighbors in the list `neighbors`
@propagate_inbounds function mapreduce_leaf(f, op, reduced, neighbors, neighbor_coords,
                                            neighborhood_search, point, point_coords,
                                            search_radius)
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

        pos_diff,
        distance2 = compute_periodic_distance(pos_diff, distance2,
                                              search_radius, periodic_box)

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
    # Use root cells of the size of the search radius
    cell_list = copy_cell_list(nhs.cell_list, search_radius)

    return TreeNeighborhoodSearch{ndims(nhs)}(; search_radius, cell_list,
                                              periodic_box = nhs.periodic_box,
                                              update_strategy = nhs.update_strategy)
end
