@doc raw"""
    TrivialNeighborhoodSearch{NDIMS}(search_radius, eachparticle, periodic_box = nothing)

Trivial neighborhood search that simply loops over all particles.
The search radius still needs to be passed in order to sort out particles outside the
search radius in the internal function `for_particle_neighbor`, but it's not used in the
internal function `eachneighbor`.

# Arguments
- `NDIMS`:          Number of dimensions.
- `search_radius`:  The uniform search radius.
- `eachparticle`:   Iterator for all particle indices. Usually just `1:n_particles`.

# Keywords
- `periodic_box = nothing`: In order to use a (rectangular) periodic domain, pass a
                            [`PeriodicBox`](@ref).
"""
struct TrivialNeighborhoodSearch{NDIMS, ELTYPE, EP, PB} <: AbstractNeighborhoodSearch
    search_radius :: ELTYPE
    eachparticle  :: EP
    periodic_box  :: PB

    function TrivialNeighborhoodSearch{NDIMS}(search_radius, eachparticle;
                                              periodic_box = nothing) where {NDIMS}
        new{NDIMS, typeof(search_radius),
            typeof(eachparticle), typeof(periodic_box)}(search_radius, eachparticle,
                                                        periodic_box)
    end
end

@inline Base.ndims(::TrivialNeighborhoodSearch{NDIMS}) where {NDIMS} = NDIMS

@inline initialize!(search::TrivialNeighborhoodSearch, x, y) = search

@inline function update!(search::TrivialNeighborhoodSearch, x, y;
                         particles_moving = (true, true))
    return search
end

@inline eachneighbor(coords, search::TrivialNeighborhoodSearch) = search.eachparticle

# Create a copy of a neighborhood search but with a different search radius
function copy_neighborhood_search(nhs::TrivialNeighborhoodSearch, search_radius, x, y)
    return nhs
end
