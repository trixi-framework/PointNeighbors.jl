"""
    CombinedGridNeighborhoodSearch(neighborhood_search::GridNeighborhoodSearch,
                                   search_radius, n_points)

Neighborhood search for multiple point sets that all use the same search radius.
Each point set has its own cell list, but all cell lists use the same grid.
`n_points` is a tuple with the number of points of each point set.
`neighborhood_search` is used as a template to create individual neighborhood searches
for each point set with [`copy_neighborhood_search`](@ref).

The cell lists are managed per point set.
Use [`initialize!`](@ref) and [`update!`](@ref) with the index of a point set to
initialize or update the cell list of this point set.
Therefore, the cell list of a point set only has to be updated once when its points move,
no matter with how many other point sets it is used.

To find neighbors, use [`grid_nhs_pair`](@ref) to obtain a neighborhood search for a pair
of point sets, which can be passed to [`foreach_point_neighbor`](@ref),
[`foreach_point`](@ref) and the other neighbor loops.
"""
struct CombinedGridNeighborhoodSearch{NHS <: GridNeighborhoodSearch}
    # One neighborhood search per point set. These are all created from the same template
    # with the same search radius, so they all use the same grid and are of the same type.
    searches::Vector{NHS}
end

function CombinedGridNeighborhoodSearch(neighborhood_search::GridNeighborhoodSearch,
                                        search_radius, n_points)
    searches = [copy_neighborhood_search(neighborhood_search, search_radius, n)
                for n in n_points]

    return CombinedGridNeighborhoodSearch(searches)
end

function Adapt.adapt_structure(to, nhs::CombinedGridNeighborhoodSearch)
    return CombinedGridNeighborhoodSearch(map(search -> Adapt.adapt(to, search),
                                              nhs.searches))
end

@inline function search_radius(nhs::CombinedGridNeighborhoodSearch)
    # All searches have the same search radius.
    return search_radius(first(nhs.searches))
end

"""
    initialize!(nhs::CombinedGridNeighborhoodSearch, point_set::Integer, coords;
                parallelization_backend = default_backend(coords),
                eachindex = axes(coords, 2))

Initialize the cell list of the point set with index `point_set` with the coordinates
`coords`. Optionally, only the points in `eachindex` are added to the cell list.
"""
function initialize!(nhs::CombinedGridNeighborhoodSearch, point_set::Integer, coords;
                     parallelization_backend = default_backend(coords),
                     eachindex = axes(coords, 2))
    initialize_grid!(nhs.searches[point_set], coords; parallelization_backend,
                     eachindex_y = eachindex)

    return nhs
end

"""
    update!(nhs::CombinedGridNeighborhoodSearch, point_set::Integer, coords;
            parallelization_backend = default_backend(coords),
            eachindex = axes(coords, 2))

Update the cell list of the point set with index `point_set` with the coordinates
`coords`. Optionally, only the points in `eachindex` are added to the cell list.
"""
function update!(nhs::CombinedGridNeighborhoodSearch, point_set::Integer, coords;
                 parallelization_backend = default_backend(coords),
                 eachindex = axes(coords, 2))
    update_grid!(nhs.searches[point_set], coords; parallelization_backend,
                 eachindex_y = eachindex)

    return nhs
end

"""
    grid_nhs_pair(nhs::CombinedGridNeighborhoodSearch,
                  point_set::Integer, neighbor_set::Integer)

Neighborhood search to find the neighbors of the points of the point set `point_set`
in the point set `neighbor_set` of `nhs`.
This can be passed to all neighbor loops, e.g., [`foreach_point_neighbor`](@ref) and
[`foreach_point`](@ref).

The returned neighborhood search is a cheap view of the cell lists of `nhs`.
It can only be used to find neighbors and cannot be initialized or updated.
Instead, initialize or update the cell lists of the point sets of `nhs`.

Note that this neighborhood search cannot be used to find neighbors of arbitrary points.
The `point_coords` array passed to the neighbor loops must be identical to the
coordinates of the point set `point_set` that were used in `initialize!` or `update!`.
"""
@inline function grid_nhs_pair(nhs::CombinedGridNeighborhoodSearch, point_set,
                                neighbor_set)
    (; searches) = nhs

    return GridNHSPair(searches[point_set], searches[neighbor_set])
end

# View of the NHSs of two point sets of a `CombinedGridNeighborhoodSearch`.
# The regular neighbor loop only uses `neighbor_search`.
# Because both cell lists use the same grid, optimized NHS implementations can also
# traverse the system points cell by cell with `system_search`.
struct GridNHSPair{NHS} <: AbstractNeighborhoodSearch
    system_search   :: NHS
    neighbor_search :: NHS
end

@inline Base.ndims(nhs::GridNHSPair) = ndims(nhs.neighbor_search)
@inline search_radius(nhs::GridNHSPair) = search_radius(nhs.neighbor_search)

# Since the `system_search` can be used to traverse the system points cell by cell,
# we have to update it if the system points move.
@inline requires_update(::GridNHSPair) = (true, true)

function initialize!(nhs::GridNHSPair, x::AbstractMatrix, y::AbstractMatrix; kwargs...)
    throw(ArgumentError("a `GridNHSPair` cannot be initialized. " *
                        "Initialize the `CombinedGridNeighborhoodSearch` instead."))
end

function update!(nhs::GridNHSPair, x::AbstractMatrix, y::AbstractMatrix; kwargs...)
    throw(ArgumentError("a `GridNHSPair` cannot be updated. " *
                        "Update the `CombinedGridNeighborhoodSearch` instead."))
end

# # Create a standalone neighborhood search like the one of the neighbor point set.
# function copy_neighborhood_search(nhs::GridNHSPair, search_radius, n_points;
#                                   eachpoint = 1:n_points)
#     return copy_neighborhood_search(nhs.neighbor_search, search_radius, n_points;
#                                     eachpoint)
# end

# The regular neighbor loop only uses the cell list of the neighbor point set.
@propagate_inbounds function mapreduce_neighbor_inner(f, op, neighbor_coords,
                                                      neighborhood_search::GridNHSPair,
                                                      point, point_coords,
                                                      search_radius, init)
    return mapreduce_neighbor_inner(f, op, neighbor_coords,
                                    neighborhood_search.neighbor_search, point,
                                    point_coords, search_radius, init)
end
