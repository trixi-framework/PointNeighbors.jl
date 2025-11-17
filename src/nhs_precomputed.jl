@doc raw"""
    PrecomputedNeighborhoodSearch{NDIMS}(; search_radius = 0.0, n_points = 0,
                                         periodic_box = nothing, update_strategy = nothing,
                                         update_neighborhood_search = GridNeighborhoodSearch{NDIMS}(),
                                         backend = DynamicVectorOfVectors{Int32},
                                         max_neighbors = max_neighbors(NDIMS))

Neighborhood search with precomputed neighbor lists. A list of all neighbors is computed
for each point during initialization and update.
This neighborhood search maximizes the performance of neighbor loops at the cost of a much
slower [`update!`](@ref).

A [`GridNeighborhoodSearch`](@ref) is used internally to compute the neighbor lists during
initialization and update.

When used on the GPU, use `freeze_neighborhood_search` after the initialization
to strip the internal neighborhood search, which is not needed anymore.

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
                            for available options. This is only used for the default value
                            of `update_neighborhood_search` below.
- `update_neighborhood_search = GridNeighborhoodSearch{NDIMS}(; periodic_box, update_strategy)`:
                            The neighborhood search used to compute the neighbor lists.
                            By default, a [`GridNeighborhoodSearch`](@ref) is used.
                            If the precomputed NHS is to be used on the GPU, make sure to
                            either freeze it after initialization and never update it again,
                            or pass a GPU-compatible neighborhood search here.
- `backend = DynamicVectorOfVectors{Int32}`: Type of the data structure to store
    the neighbor lists. Can be
    - `Vector{Vector{Int32}}`: Scattered memory, but very memory-efficient.
    - `DynamicVectorOfVectors{Int32}`: Contiguous memory, optimizing cache-hits
                                       and GPU-compatible.
- `max_neighbors`: Maximum number of neighbors per particle. This will be used to
                   allocate the `DynamicVectorOfVectors`. It is not used with
                   other backends. The default is 64 in 2D and 324 in 3D.
"""
struct PrecomputedNeighborhoodSearch{NDIMS, NL, ELTYPE, PB, NHS} <:
       AbstractNeighborhoodSearch
    neighbor_lists      :: NL
    search_radius       :: ELTYPE
    periodic_box        :: PB
    neighborhood_search :: NHS

    function PrecomputedNeighborhoodSearch{NDIMS}(neighbor_lists, search_radius,
                                                  periodic_box,
                                                  update_neighborhood_search) where {NDIMS}
        return new{NDIMS, typeof(neighbor_lists),
                   typeof(search_radius),
                   typeof(periodic_box),
                   typeof(update_neighborhood_search)}(neighbor_lists, search_radius,
                                                       periodic_box,
                                                       update_neighborhood_search)
    end
end

function PrecomputedNeighborhoodSearch{NDIMS}(; search_radius = 0.0, n_points = 0,
                                              periodic_box = nothing,
                                              update_strategy = nothing,
                                              update_neighborhood_search = GridNeighborhoodSearch{NDIMS}(;
                                                                                                         search_radius,
                                                                                                         n_points,
                                                                                                         periodic_box,
                                                                                                         update_strategy),
                                              backend = DynamicVectorOfVectors{Int32},
                                              max_neighbors = max_neighbors(NDIMS)) where {NDIMS}
    neighbor_lists = construct_backend(backend, n_points, max_neighbors)

    PrecomputedNeighborhoodSearch{NDIMS}(neighbor_lists, search_radius,
                                         periodic_box, update_neighborhood_search)
end

# Default values for maximum neighbor count
function max_neighbors(NDIMS)
    if NDIMS == 1
        return 32
    elseif NDIMS == 2
        return 64
    elseif NDIMS == 3
        return 320
    end

    throw(ArgumentError("`NDIMS` must be 1, 2, or 3"))
end

@inline Base.ndims(::PrecomputedNeighborhoodSearch{NDIMS}) where {NDIMS} = NDIMS

@inline requires_update(::PrecomputedNeighborhoodSearch) = (true, true)

function initialize!(search::PrecomputedNeighborhoodSearch,
                     x::AbstractMatrix, y::AbstractMatrix;
                     parallelization_backend = default_backend(x),
                     eachindex_y = axes(y, 2))
    (; neighborhood_search, neighbor_lists) = search

    if eachindex_y != axes(y, 2)
        error("this neighborhood search does not support inactive points")
    end

    # Initialize grid NHS
    initialize!(neighborhood_search, x, y; parallelization_backend)

    initialize_neighbor_lists!(neighbor_lists, neighborhood_search, x, y,
                               parallelization_backend)

    return search
end

function update!(search::PrecomputedNeighborhoodSearch,
                 x::AbstractMatrix, y::AbstractMatrix;
                 points_moving = (true, true), parallelization_backend = default_backend(x),
                 eachindex_y = axes(y, 2))
    (; neighborhood_search, neighbor_lists) = search

    if eachindex_y != axes(y, 2)
        error("this neighborhood search does not support inactive points")
    end

    # Update the internal neighborhood search
    update!(neighborhood_search, x, y; points_moving, parallelization_backend)

    # Skip update if both point sets are static
    if any(points_moving)
        initialize_neighbor_lists!(neighbor_lists, neighborhood_search, x, y,
                                   parallelization_backend)
    end

    return search
end

function initialize_neighbor_lists!(neighbor_lists, neighborhood_search, x, y,
                                    parallelization_backend)
    # Initialize neighbor lists
    empty!(neighbor_lists)
    resize!(neighbor_lists, size(x, 2))
    for i in eachindex(neighbor_lists)
        neighbor_lists[i] = Int[]
    end

    # Fill neighbor lists
    foreach_point_neighbor(x, y, neighborhood_search;
                           parallelization_backend) do point, neighbor, _, _
        push!(neighbor_lists[point], neighbor)
    end
end

function initialize_neighbor_lists!(neighbor_lists::DynamicVectorOfVectors,
                                    neighborhood_search, x, y, parallelization_backend)
    resize!(neighbor_lists, size(x, 2))

    # `Base.empty!.(neighbor_lists)`, but for all backends
    @threaded parallelization_backend for i in eachindex(neighbor_lists)
        emptyat!(neighbor_lists, i)
    end

    # Fill neighbor lists
    foreach_point_neighbor(x, y, neighborhood_search;
                           parallelization_backend) do point, neighbor, _, _
        pushat!(neighbor_lists, point, neighbor)
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

        pos_diff,
        distance2 = compute_periodic_distance(pos_diff, distance2, search_radius,
                                              periodic_box)

        distance = sqrt(distance2)

        # Inline to avoid loss of performance
        # compared to not using `foreach_point_neighbor`.
        @inline f(point, neighbor, pos_diff, distance)
    end
end

function copy_neighborhood_search(nhs::PrecomputedNeighborhoodSearch,
                                  search_radius, n_points; eachpoint = 1:n_points)
    update_neighborhood_search = copy_neighborhood_search(nhs.neighborhood_search,
                                                          search_radius, n_points;
                                                          eachpoint)

    # For `Vector{Vector}` backend use `max_neighbors(NDIMS)` as fallback.
    # This should never be used because this backend doesn't require a `max_neighbors`.
    max_neighbors_ = max_inner_length(nhs.neighbor_lists, max_neighbors(ndims(nhs)))
    return PrecomputedNeighborhoodSearch{ndims(nhs)}(; search_radius, n_points,
                                                     periodic_box = nhs.periodic_box,
                                                     update_neighborhood_search,
                                                     backend = typeof(nhs.neighbor_lists),
                                                     max_neighbors = max_neighbors_)
end

@inline function freeze_neighborhood_search(search::PrecomputedNeighborhoodSearch)
    # Indicate that the neighborhood search is static and will not be updated anymore.
    # For the `PrecomputedNeighborhoodSearch`, strip the inner neighborhood search,
    # which is used only for initialization and updating.
    return PrecomputedNeighborhoodSearch{ndims(search)}(search.neighbor_lists,
                                                        search.search_radius,
                                                        search.periodic_box,
                                                        nothing)
end
