@doc raw"""
    TrivialNeighborhoodSearch{NDIMS}(search_radius, eachparticle)

Trivial neighborhood search that simply loops over all particles.
The search radius still needs to be passed in order to sort out particles outside the
search radius in the internal function `for_particle_neighbor`, but it's not used in the
internal function `eachneighbor`.

# Arguments
- `NDIMS`:          Number of dimensions.
- `search_radius`:  The uniform search radius.
- `eachparticle`:   `UnitRange` of all particle indices. Usually just `1:n_particles`.

# Keywords
- `periodic_box_min_corner`:    In order to use a (rectangular) periodic domain, pass the
                                coordinates of the domain corner in negative coordinate
                                directions.
- `periodic_box_max_corner`:    In order to use a (rectangular) periodic domain, pass the
                                coordinates of the domain corner in positive coordinate
                                directions.
"""
struct TrivialNeighborhoodSearch{NDIMS, ELTYPE, EP, PB}
    search_radius :: ELTYPE
    eachparticle  :: EP
    periodic_box  :: PB

    function TrivialNeighborhoodSearch{NDIMS}(search_radius, eachparticle;
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
            typeof(eachparticle), typeof(periodic_box)}(search_radius, eachparticle,
                                                        periodic_box)
    end
end

@inline function Base.ndims(neighborhood_search::TrivialNeighborhoodSearch{NDIMS}) where {
                                                                                          NDIMS
                                                                                          }
    return NDIMS
end

@inline initialize!(search::TrivialNeighborhoodSearch, coords_fun) = search
@inline update!(search::TrivialNeighborhoodSearch, coords_fun) = search
@inline eachneighbor(coords, search::TrivialNeighborhoodSearch) = search.eachparticle
