@doc raw"""
    TrivialNeighborhoodSearch{NDIMS}(; search_radius = 0.0, eachpoint = 1:0,
                                     periodic_box = nothing)

Trivial neighborhood search that simply loops over all points.

# Arguments
- `NDIMS`: Number of dimensions.

# Keywords
- `search_radius = 0.0`:    The fixed search radius. The default of `0.0` is useful together
                            with [`copy_neighborhood_search`](@ref).
- `eachpoint = 1:0`:        Iterator for all point indices. Usually just `1:n_points`.
                            The default of `1:0` is useful together with
                            [`copy_neighborhood_search`](@ref).
- `periodic_box = nothing`: In order to use a (rectangular) periodic domain, pass a
                            [`PeriodicBox`](@ref).
"""
struct TrivialNeighborhoodSearch{NDIMS, ELTYPE, EP, PB} <: AbstractNeighborhoodSearch
    search_radius :: ELTYPE
    eachpoint     :: EP
    periodic_box  :: PB

    function TrivialNeighborhoodSearch{NDIMS}(; search_radius = 0.0, eachpoint = 1:0,
                                              periodic_box = nothing) where {NDIMS}
        new{NDIMS, typeof(search_radius),
            typeof(eachpoint), typeof(periodic_box)}(search_radius, eachpoint, periodic_box)
    end
end

@inline Base.ndims(::TrivialNeighborhoodSearch{NDIMS}) where {NDIMS} = NDIMS

@inline initialize!(search::TrivialNeighborhoodSearch, x, y) = search

@inline function update!(search::TrivialNeighborhoodSearch, x, y;
                         points_moving = (true, true))
    return search
end

@inline eachneighbor(coords, search::TrivialNeighborhoodSearch) = search.eachpoint

# Create a copy of a neighborhood search but with a different search radius
function copy_neighborhood_search(nhs::TrivialNeighborhoodSearch, search_radius, x, y)
    return nhs
end

function copy_neighborhood_search(nhs::TrivialNeighborhoodSearch,
                                  search_radius, n_points; eachpoint = 1:n_points)
    return TrivialNeighborhoodSearch{ndims(nhs)}(; search_radius, eachpoint,
                                                 periodic_box = nhs.periodic_box)
end
