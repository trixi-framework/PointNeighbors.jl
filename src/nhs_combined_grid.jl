"""
    CombinedGridNeighborhoodSearch(neighborhood_search::GridNeighborhoodSearch,
                                   search_radius, n_points)

Neighborhood search for multiple point sets with the same search radius.
Each point set has its own cell list, but all cell lists use the same grid.
`n_points` is a collection with the number of points of each point set.
The neighborhood searches of the point sets are created from the template
`neighborhood_search` with [`copy_neighborhood_search`](@ref).
The coordinates of all point sets must have the same type.

Use [`initialize!`](@ref) and [`update!`](@ref) with the index of a point set to
initialize or update the cell list of this point set.
Use [`point_set_pair`](@ref) to obtain a neighborhood search to find neighbors
of the points of one point set in another point set.

Because all point sets are stored in the same grid, the cell lists of two point sets
can be used to compute the neighbor loop of [`foreach_point`](@ref) with
`simd = Val(true)` cell by cell, both for two different point sets and for a point set
with itself.
See [`foreach_point`](@ref) for the requirements of the SIMD-vectorized neighbor loop.
"""
struct CombinedGridNeighborhoodSearch{NHS}
    searches::Vector{NHS}

    function CombinedGridNeighborhoodSearch(searches::Vector{NHS}) where {NHS}
        return new{NHS}(searches)
    end
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

@inline search_radius(nhs::CombinedGridNeighborhoodSearch) = search_radius(first(nhs.searches))

"""
    initialize!(nhs::CombinedGridNeighborhoodSearch, point_set, coords;
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
    update!(nhs::CombinedGridNeighborhoodSearch, point_set, coords;
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
    point_set_pair(nhs::CombinedGridNeighborhoodSearch, system_index, neighbor_index)

Neighborhood search to find the neighbors of the points of the point set `system_index`
in the point set `neighbor_index` of `nhs`.
The returned neighborhood search shares the cell lists with `nhs`.
It cannot be initialized or updated itself. Instead, initialize or update the cell lists
of the point sets of `nhs`.

When calling [`foreach_point`](@ref) with `simd = Val(true)` with this neighborhood search,
the cell list of the point set `system_index` must have been initialized or updated with
`system_coords`, and the cell list of the point set `neighbor_index` with `neighbor_coords`.
The `eachindex` of the cell list of the point set `system_index` must contain all `points`
passed to [`foreach_point`](@ref).

When the cell lists sort the points (see [`sorts_points`](@ref)),
[`point_permutation`](@ref) of the returned neighborhood search is the permutation of the
point set `neighbor_index`.
[`sorted_neighborhood_search`](@ref) of the returned neighborhood search can only be used
when the points of both point sets are reordered with the permutations of their own
cell lists, that is, with `point_permutation(point_set_pair(nhs, i, i))` for point set `i`.
"""
@inline function point_set_pair(nhs::CombinedGridNeighborhoodSearch, system_index,
                                neighbor_index)
    (; searches) = nhs

    return PointSetPairSearch(searches[system_index], searches[neighbor_index],
                              Int(system_index), Int(neighbor_index))
end

# Neighborhood search to find the neighbors of the points of one point set
# (in the cell list of `system_search`) in another point set (in the cell list of
# `neighbor_search`). Both cell lists use the same grid.
# The regular neighbor loop only uses `neighbor_search`.
struct PointSetPairSearch{NHS, ELTYPE, PB} <: AbstractNeighborhoodSearch
    system_search   :: NHS
    neighbor_search :: NHS
    system_index    :: Int
    neighbor_index  :: Int
    search_radius   :: ELTYPE
    periodic_box    :: PB

    function PointSetPairSearch(system_search::NHS, neighbor_search::NHS, system_index,
                                neighbor_index) where {NHS}
        (; search_radius, periodic_box) = neighbor_search

        return new{NHS, typeof(search_radius),
                   typeof(periodic_box)}(system_search, neighbor_search, system_index,
                                         neighbor_index, search_radius, periodic_box)
    end
end

function Adapt.adapt_structure(to, nhs::PointSetPairSearch)
    return PointSetPairSearch(Adapt.adapt(to, nhs.system_search),
                              Adapt.adapt(to, nhs.neighbor_search), nhs.system_index,
                              nhs.neighbor_index)
end

@inline Base.ndims(nhs::PointSetPairSearch) = ndims(nhs.neighbor_search)

# Moving points in `x` require an update of the cell list of the system point set,
# which is used by the SIMD-vectorized neighbor loop.
@inline requires_update(::PointSetPairSearch) = (true, true)

# The cell lists are shared with the `CombinedGridNeighborhoodSearch`,
# so they must be initialized and updated there, once for each point set.
function initialize!(nhs::PointSetPairSearch, x::AbstractMatrix, y::AbstractMatrix;
                     kwargs...)
    throw(ArgumentError("a `PointSetPairSearch` cannot be initialized. " *
                        "Initialize the `CombinedGridNeighborhoodSearch` instead."))
end

function update!(nhs::PointSetPairSearch, x::AbstractMatrix, y::AbstractMatrix; kwargs...)
    throw(ArgumentError("a `PointSetPairSearch` cannot be updated. " *
                        "Update the `CombinedGridNeighborhoodSearch` instead."))
end

# The permutation sorts the points of the neighbor point set.
@inline sorts_points(nhs::PointSetPairSearch) = sorts_points(nhs.neighbor_search)

@inline point_permutation(nhs::PointSetPairSearch) = point_permutation(nhs.neighbor_search)

# The returned neighborhood search requires the points of both point sets to be reordered
# with the permutations of their own cell lists.
function sorted_neighborhood_search(nhs::PointSetPairSearch)
    return PointSetPairSearch(sorted_neighborhood_search(nhs.system_search),
                              sorted_neighborhood_search(nhs.neighbor_search),
                              nhs.system_index, nhs.neighbor_index)
end

function copy_neighborhood_search(nhs::PointSetPairSearch, search_radius, n_points;
                                  eachpoint = 1:n_points)
    return copy_neighborhood_search(nhs.neighbor_search, search_radius, n_points;
                                    eachpoint)
end

# The regular neighbor loop only uses the cell list of the neighbor point set.
@propagate_inbounds function mapreduce_neighbor_inner(f, op, neighbor_coords,
                                                      neighborhood_search::PointSetPairSearch,
                                                      point, point_coords,
                                                      search_radius, init)
    return mapreduce_neighbor_inner(f, op, neighbor_coords,
                                    neighborhood_search.neighbor_search, point,
                                    point_coords, search_radius, init)
end
