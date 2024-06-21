@doc raw"""
    TrivialNeighborhoodSearch{NDIMS}(search_radius, eachpoint)

Trivial neighborhood search that simply loops over all points.

# Arguments
- `NDIMS`:          Number of dimensions.
- `search_radius`:  The uniform search radius.
- `eachpoint`:   `UnitRange` of all point indices. Usually just `1:n_points`.

# Keywords
- `periodic_box_min_corner`:    In order to use a (rectangular) periodic domain, pass the
                                coordinates of the domain corner in negative coordinate
                                directions.
- `periodic_box_max_corner`:    In order to use a (rectangular) periodic domain, pass the
                                coordinates of the domain corner in positive coordinate
                                directions.
"""
struct TrivialNeighborhoodSearch{NDIMS, ELTYPE, EP, PB} <: AbstractNeighborhoodSearch
    search_radius :: ELTYPE
    eachpoint     :: EP
    periodic_box  :: PB

    function TrivialNeighborhoodSearch{NDIMS}(search_radius, eachpoint;
                                              periodic_box_min_corner = nothing,
                                              periodic_box_max_corner = nothing) where {
                                                                                        NDIMS
                                                                                        }
        if search_radius < eps() ||
           (periodic_box_min_corner === nothing && periodic_box_max_corner === nothing)

            # No periodicity
            periodic_box = nothing
        elseif periodic_box_min_corner !== nothing && periodic_box_max_corner !== nothing
            periodic_box = PeriodicBox(periodic_box_min_corner, periodic_box_max_corner)
        else
            throw(ArgumentError("`periodic_box_min_corner` and `periodic_box_max_corner` " *
                                "must either be both `nothing` or both an array or tuple"))
        end

        new{NDIMS, typeof(search_radius),
            typeof(eachpoint), typeof(periodic_box)}(search_radius, eachpoint, periodic_box)
    end
end

@inline function Base.ndims(neighborhood_search::TrivialNeighborhoodSearch{NDIMS}) where {
                                                                                          NDIMS
                                                                                          }
    return NDIMS
end

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
