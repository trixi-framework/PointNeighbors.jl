@doc raw"""
    PrecomputedNeighborhoodSearch{NDIMS}(; search_radius = 0.0, n_points = 0,
                                         periodic_box = nothing, update_strategy = nothing)

Neighborhood search with precomputed neighbor lists. A list of all neighbors is computed
for each point during initialization and update.
This neighborhood search maximizes the performance of neighbor loops at the cost of a much
slower [`update!`](@ref).

A [`GridNeighborhoodSearch`](@ref) is used internally to compute the neighbor lists during
initialization and update.

# Arguments
- `NDIMS`: Number of dimensions.

# Keywords
- `search_radius = 0.0`:    The fixed search radius. The default of `0.0` is useful together
                            with [`copy_neighborhood_search`](@ref).
- `n_points = 0`:           Total number of points. The default of `0` is useful together
                            with [`copy_neighborhood_search`](@ref).
- `periodic_box = nothing`: In order to use a (rectangular) periodic domain, pass a
                            [`PeriodicBox`](@ref).
- `update_strategy`:        Strategy to parallelize `update!` of the internally used
                            `GridNeighborhoodSearch`. See [`GridNeighborhoodSearch`](@ref)
                            for available options.
"""
struct PrecomputedNeighborhoodSearch{NDIMS, NHS, NL, PB} <: AbstractNeighborhoodSearch
    neighborhood_search :: NHS
    neighbor_lists      :: NL
    periodic_box        :: PB

    function PrecomputedNeighborhoodSearch{NDIMS}(; search_radius = 0.0, n_points = 0,
                                                  periodic_box = nothing,
                                                  update_strategy = nothing) where {NDIMS}
        nhs = GridNeighborhoodSearch{NDIMS}(; search_radius, n_points,
                                            periodic_box, update_strategy)

        neighbor_lists = Vector{Vector{Int}}()

        new{NDIMS, typeof(nhs),
            typeof(neighbor_lists),
            typeof(periodic_box)}(nhs, neighbor_lists, periodic_box)
    end
end

@inline Base.ndims(::PrecomputedNeighborhoodSearch{NDIMS}) where {NDIMS} = NDIMS

@inline requires_update(::PrecomputedNeighborhoodSearch) = (true, true)

@inline function search_radius(search::PrecomputedNeighborhoodSearch)
    return search_radius(search.neighborhood_search)
end

function initialize!(search::PrecomputedNeighborhoodSearch,
                     x::AbstractMatrix, y::AbstractMatrix)
    (; neighborhood_search, neighbor_lists) = search

    # Initialize grid NHS
    initialize!(neighborhood_search, x, y)

    initialize_neighbor_lists!(neighbor_lists, neighborhood_search, x, y)
end

# WARNING! Experimental feature:
# By default, determine the parallelization backend from the type of `x`.
# Optionally, pass a `KernelAbstractions.Backend` to run the KernelAbstractions.jl code
# on this backend. This can be useful to run GPU kernels on the CPU by passing
# `parallelization_backend = KernelAbstractions.CPU()`, even though `x isa Array`.
function update!(search::PrecomputedNeighborhoodSearch,
                 x::AbstractMatrix, y::AbstractMatrix;
                 points_moving = (true, true), parallelization_backend = x)
    (; neighborhood_search, neighbor_lists) = search

    # Update grid NHS
    update!(neighborhood_search, x, y; points_moving, parallelization_backend)

    # Skip update if both point sets are static
    if any(points_moving)
        initialize_neighbor_lists!(neighbor_lists, neighborhood_search, x, y)
    end
end

function initialize_neighbor_lists!(neighbor_lists, neighborhood_search, x, y)
    # Initialize neighbor lists
    empty!(neighbor_lists)
    resize!(neighbor_lists, size(x, 2))
    for i in eachindex(neighbor_lists)
        neighbor_lists[i] = Int[]
    end

    # Fill neighbor lists
    foreach_point_neighbor(x, y, neighborhood_search) do point, neighbor, _, _
        push!(neighbor_lists[point], neighbor)
    end
end

@inline function foreach_neighbor(f, neighbor_system_coords,
                                  neighborhood_search::PrecomputedNeighborhoodSearch,
                                  point, point_coords, search_radius)
    (; periodic_box, neighbor_lists) = neighborhood_search

    neighbors = @inbounds neighbor_lists[point]
    for neighbor_ in eachindex(neighbors)
        neighbor = @inbounds neighbors[neighbor_]

        # Making the following `@inbounds` yields a ~2% speedup on an NVIDIA H100.
        # But we don't know if `neighbor` (extracted from the cell list) is in bounds.
        neighbor_coords = extract_svector(neighbor_system_coords,
                                          Val(ndims(neighborhood_search)), neighbor)

        pos_diff = point_coords - neighbor_coords
        distance2 = dot(pos_diff, pos_diff)

        pos_diff, distance2 = compute_periodic_distance(pos_diff, distance2, search_radius,
                                                        periodic_box)

        distance = sqrt(distance2)

        # Inline to avoid loss of performance
        # compared to not using `foreach_point_neighbor`.
        @inline f(point, neighbor, pos_diff, distance)
    end
end

function copy_neighborhood_search(nhs::PrecomputedNeighborhoodSearch,
                                  search_radius, n_points; eachpoint = 1:n_points)
    update_strategy_ = nhs.neighborhood_search.update_strategy
    return PrecomputedNeighborhoodSearch{ndims(nhs)}(; search_radius, n_points,
                                                     periodic_box = nhs.periodic_box,
                                                     update_strategy = update_strategy_)
end
