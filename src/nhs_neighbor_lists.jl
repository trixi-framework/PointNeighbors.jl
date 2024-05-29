struct NeighborListsNeighborhoodSearch{NDIMS, NHS, NL, PB}
    neighborhood_search :: NHS
    neighbor_lists      :: NL
    periodic_box        :: PB

    function NeighborListsNeighborhoodSearch{NDIMS}(search_radius, n_particles;
                                                    periodic_box_min_corner = nothing,
                                                    periodic_box_max_corner = nothing) where {
                                                                                              NDIMS
                                                                                              }
        nhs = GridNeighborhoodSearch{NDIMS}(search_radius, n_particles,
                                            periodic_box_min_corner = periodic_box_min_corner,
                                            periodic_box_max_corner = periodic_box_max_corner)

        neighbor_lists = Vector{Vector{Int}}()

        new{NDIMS, typeof(nhs),
            typeof(neighbor_lists),
            typeof(nhs.periodic_box)}(nhs, neighbor_lists, nhs.periodic_box)
    end
end

@inline function Base.ndims(neighborhood_search::NeighborListsNeighborhoodSearch{NDIMS}) where {
                                                                                                NDIMS
                                                                                                }
    return NDIMS
end

function initialize!(search::NeighborListsNeighborhoodSearch,
                     x::AbstractMatrix, y::AbstractMatrix)
    (; neighborhood_search, neighbor_lists) = search

    initialize!(neighborhood_search, x, y)

    initialize_neighbor_lists!(neighbor_lists, neighborhood_search, x, y)
end

function update!(search::NeighborListsNeighborhoodSearch,
                 x::AbstractMatrix, y::AbstractMatrix;
                 particles_moving = (true, true))
    (; neighborhood_search, neighbor_lists) = search

    update!(neighborhood_search, x, y, particles_moving = particles_moving)

    initialize_neighbor_lists!(neighbor_lists, neighborhood_search, x, y)
end

function initialize_neighbor_lists!(neighbor_lists, neighborhood_search, x, y)
    # Initialize neighbor lists
    empty!(neighbor_lists)
    resize!(neighbor_lists, size(x, 2))
    for i in eachindex(neighbor_lists)
        neighbor_lists[i] = Int[]
    end

    # Compute neighbor lists
    for_particle_neighbor(x, y, neighborhood_search) do particle, neighbor, _, _
        push!(neighbor_lists[particle], neighbor)
    end
end

@inline function for_particle_neighbor_inner(f, system_coords, neighbor_system_coords,
                                             neighborhood_search::NeighborListsNeighborhoodSearch,
                                             particle)
    (; periodic_box, neighbor_lists) = neighborhood_search
    (; search_radius) = neighborhood_search.neighborhood_search

    particle_coords = extract_svector(system_coords, Val(ndims(neighborhood_search)),
                                      particle)
    for neighbor in neighbor_lists[particle]
        neighbor_coords = extract_svector(neighbor_system_coords,
                                          Val(ndims(neighborhood_search)), neighbor)

        pos_diff = particle_coords - neighbor_coords
        distance2 = dot(pos_diff, pos_diff)

        pos_diff, distance2 = compute_periodic_distance(pos_diff, distance2, search_radius,
                                                        periodic_box)

        distance = sqrt(distance2)

        # Inline to avoid loss of performance
        # compared to not using `for_particle_neighbor`.
        @inline f(particle, neighbor, pos_diff, distance)
    end
end
