@doc raw"""
    PrecomputedNeighborhoodSearch{NDIMS}(; search_radius = 0.0, n_points = 0,
                                         periodic_box = nothing, update_strategy = nothing,
                                         update_neighborhood_search = GridNeighborhoodSearch{NDIMS}(),
                                         update_neighborhood_search_padding = 0.05,
                                         backend = DynamicVectorOfVectors{Int32},
                                         transpose_backend = false,
                                         max_neighbors = max_neighbors(NDIMS))
                                         sort_neighbor_lists = true)

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
                            Note that the type of `search_radius` determines the type used
                            for the distance computations.
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
- `update_neighborhood_search_padding = 0.05`: Relative padding used for the fixed
                            search radius of the internal [`GridNeighborhoodSearch`](@ref)
                            that computes the neighbor lists.
- `backend = DynamicVectorOfVectors{Int32}`: Type of the data structure to store
    the neighbor lists. Can be
    - `Vector{Vector{Int32}}`: Scattered memory, but very memory-efficient.
    - `DynamicVectorOfVectors{Int32}`: Contiguous memory, optimizing cache-hits
                                       and GPU-compatible.
- `transpose_backend = false`: Whether to transpose the backend data structure storing the
                            neighbor lists. This is only supported for the
                            `DynamicVectorOfVectors` backend.
                            By default, the neighbors of each point are stored contiguously
                            in memory. This layout optimizes cache hits when looping
                            over all neighbors of a point on CPUs.
                            On GPUs, however, storing all first neighbors of all points
                            contiguously in memory, then all second neighbors, etc.,
                            (`transpose_backend = true`) allows for coalesced
                            memory accesses when all threads process the n-th neighbor
                            of their respective point in parallel.
                            This can lead to a speedup of ~3x in many cases.
- `max_neighbors`: Maximum number of neighbors per particle. This will be used to
                   allocate the `DynamicVectorOfVectors`. It is not used with
                   other backends. The default is 64 in 2D and 324 in 3D.
- `sort_neighbor_lists = true`: Whether to sort the neighbor lists after construction.
                            This can improve cache hits on CPUs and improve coalesced
                            memory access on GPUs.
"""
struct PrecomputedNeighborhoodSearch{NDIMS, NL, ELTYPE, PB, NHS} <:
       AbstractNeighborhoodSearch
    neighbor_lists      :: NL
    search_radius       :: ELTYPE
    periodic_box        :: PB
    neighborhood_search :: NHS
    sort_neighbor_lists :: Bool
    update_neighborhood_search_padding :: Float64

    function PrecomputedNeighborhoodSearch{NDIMS}(neighbor_lists, search_radius,
                                                  periodic_box,
                                                  update_neighborhood_search,
                                                  sort_neighbor_lists,
                                                  update_neighborhood_search_padding) where {NDIMS}
        return new{NDIMS, typeof(neighbor_lists),
                   typeof(search_radius),
                   typeof(periodic_box),
                   typeof(update_neighborhood_search)}(neighbor_lists, search_radius,
                                                       periodic_box,
                                                       update_neighborhood_search,
                                                       sort_neighbor_lists,
                                                       update_neighborhood_search_padding)
    end
end

function PrecomputedNeighborhoodSearch{NDIMS}(; search_radius = 0.0, n_points = 0,
                                              periodic_box = nothing,
                                              update_strategy = nothing,
                                              update_neighborhood_search_padding = 0.05,
                                              update_neighborhood_search = GridNeighborhoodSearch{NDIMS}(;
                                                                                                         search_radius = search_radius * (1 + update_neighborhood_search_padding),
                                                                                                         n_points,
                                                                                                         periodic_box,
                                                                                                         update_strategy),
                                              backend = DynamicVectorOfVectors{Int32},
                                              transpose_backend = false,
                                              max_neighbors = max_neighbors(NDIMS),
                                              sort_neighbor_lists = true) where {NDIMS}
    neighbor_lists = construct_backend(backend, n_points, max_neighbors; transpose_backend)

    PrecomputedNeighborhoodSearch{NDIMS}(neighbor_lists, search_radius,
                                         periodic_box, update_neighborhood_search,
                                         sort_neighbor_lists,
                                         update_neighborhood_search_padding)
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
                               search.search_radius,
                               parallelization_backend, search.sort_neighbor_lists)

    return search
end

function update!(search::PrecomputedNeighborhoodSearch,
                 x::AbstractMatrix, y::AbstractMatrix;
                 points_moving = (true, true), parallelization_backend = default_backend(x),
                 eachindex_y = axes(y, 2), search_radius = search.search_radius)
    (; neighborhood_search, neighbor_lists) = search

    if eachindex_y != axes(y, 2)
        error("this neighborhood search does not support inactive points")
    end

    # Update the internal neighborhood search
    update!(neighborhood_search, x, y; points_moving, parallelization_backend)

    # Skip update if both point sets are static
    if any(points_moving)
        initialize_neighbor_lists!(neighbor_lists, neighborhood_search, x, y,
                                   search_radius,
                                   parallelization_backend, search.sort_neighbor_lists)
    end

    return search
end

function initialize_neighbor_lists!(neighbor_lists, neighborhood_search, x, y, search_radius,
                                    parallelization_backend, sort_neighbor_lists)
    # Initialize neighbor lists
    empty!(neighbor_lists)
    resize!(neighbor_lists, size(x, 2))
    for i in eachindex(neighbor_lists)
        neighbor_lists[i] = Int[]
    end

    # Fill neighbor lists
    foreach_point_neighbor(x, y, neighborhood_search;
                           parallelization_backend) do point, neighbor, _, distance
        if distance <= search_radius
            push!(neighbor_lists[point], neighbor)
        end
    end
end

using SIMD
function initialize_neighbor_lists!(neighbor_lists::DynamicVectorOfVectors,
                                    neighborhood_search, x, y, search_radius,
                                    parallelization_backend, sort_neighbor_lists)
    resize!(neighbor_lists, size(x, 2))

    # `Base.empty!.(neighbor_lists)`, but for all backends
    @threaded parallelization_backend for i in eachindex(neighbor_lists)
        emptyat!(neighbor_lists, i)
    end

    # Fill neighbor lists
    # foreach_point_neighbor(x, y, neighborhood_search;
    #                        parallelization_backend) do point, neighbor, _, _
    #     @inbounds pushat!(neighbor_lists, point, neighbor)
    # end
    # @threaded parallelization_backend for point in axes(x, 2)
    #     length = @inbounds neighbor_lists.lengths[point]
    #     length = foreach_neighbor_unsafe(x, y, neighborhood_search, point, length) do point_, neighbor, _, distance, length_
    #         # @inbounds pushat!(neighbor_lists, point_, neighbor)
    #         if distance < search_radius(neighborhood_search)
    #             length_ += 1
    #             @inbounds neighbor_lists.backend[length_, point] = neighbor
    #         end

    #         return length_
    #     end
    #     @inbounds neighbor_lists.lengths[point] = length
    # end

    search_radius2 = search_radius^2

    # 100x100x100 points on Rucio: 40ms on the CPU, 65ms on the GPU.
    # 100x100x100 points on RAMSES: 36ms on the CPU
    @threaded parallelization_backend for point in axes(x, 2)
        point_coords = @inbounds extract_svector(x, Val(ndims(neighborhood_search)), point)
        cell = cell_coords(point_coords, neighborhood_search)
        length = @inbounds neighbor_lists.lengths[point]

        @inbounds @fastmath for neighbor_cell_ in neighboring_cells(cell, neighborhood_search)
            neighbor_cell = Tuple(neighbor_cell_)
            neighbors = points_in_cell(neighbor_cell, neighborhood_search)

            for neighbor_ in eachindex(neighbors)
                neighbor = @inbounds neighbors[neighbor_]

                neighbor_point_coords = extract_svector(y, Val(ndims(neighborhood_search)),
                                                        neighbor)

                pos_diff = convert.(eltype(neighborhood_search),
                                    point_coords - neighbor_point_coords)
                distance2 = dot(pos_diff, pos_diff)

                @inbounds neighbor_lists.backend[length + 1, point] = neighbor
                length = length + (distance2 <= search_radius2)
            end
        end

        @inbounds neighbor_lists.lengths[point] = length
    end

    # 100x100x100 points on Rucio: 83ms on the CPU, 16ms on the GPU.
    # 100x100x100 points on RAMSES: 52ms on the CPU
    # @threaded parallelization_backend for point in axes(x, 2)
    #     point_coords = @inbounds extract_svector(x, Val(ndims(neighborhood_search)), point)
    #     cell = cell_coords(point_coords, neighborhood_search)
    #     length = @inbounds neighbor_lists.lengths[point]

    #     @inbounds @fastmath for neighbor_cell_ in neighboring_cells(cell, neighborhood_search)
    #         neighbor_cell = Tuple(neighbor_cell_)
    #         neighbors = points_in_cell(neighbor_cell, neighborhood_search)

    #         for neighbor_ in eachindex(neighbors)
    #             neighbor = @inbounds neighbors[neighbor_]

    #             neighbor_point_coords = extract_svector(y, Val(ndims(neighborhood_search)),
    #                                                     neighbor)

    #             pos_diff = convert.(eltype(neighborhood_search),
    #                                 point_coords - neighbor_point_coords)
    #             distance2 = dot(pos_diff, pos_diff)

    #             if distance2 <= search_radius2
    #                 length = length + 1
    #                 @inbounds neighbor_lists.backend[length, point] = neighbor
    #             end
    #         end
    #     end

    #     @inbounds neighbor_lists.lengths[point] = length
    # end

    # 100x100x100 points on Rucio: 11ms on the GPU (fastest), not CPU-compatible.
    # ndrange = size(x, 2)
    # mykernel(parallelization_backend, 64)(x, y, neighborhood_search, neighbor_lists, search_radius2, ndrange = ndrange)
    # KernelAbstractions.synchronize(parallelization_backend)

    # 100x100x100 points on Rucio: 36ms on the CPU (fastest), not GPU-compatible.
    # 100x100x100 points on RAMSES: 38ms on the CPU
    # @threaded parallelization_backend for point in axes(x, 2)
    #     point_coords = @inbounds extract_svector(x, Val(ndims(neighborhood_search)), point)
    #     coords_a1, coords_a2, coords_a3 = point_coords
    #     cell = cell_coords(point_coords, neighborhood_search)
    #     length_ = @inbounds neighbor_lists.lengths[point]

    #     @inbounds @fastmath for neighbor_cell_ in neighboring_cells(cell, neighborhood_search)
    #         neighbor_cell = Tuple(neighbor_cell_)
    #         neighbors = points_in_cell(neighbor_cell, neighborhood_search)

    #         vectorwidth = 8
    #         @fastmath for block_start in 1:vectorwidth:length(neighbors)
    #             block_start + vectorwidth - 1 > length(neighbors) && break

    #             neighbors_block = @inbounds vload(Vec{8, eltype(neighbors)}, neighbors, block_start)

    #             # Linear indexing into `coordinates` because Cartesian indexing doesn't work.
    #             point_start = (neighbors_block - 1) * 3 + 1
    #             x_b = @inbounds y[point_start]
    #             y_b = @inbounds y[point_start + 1]
    #             z_b = @inbounds y[point_start + 2]

    #             pos_diff_x = coords_a1 - x_b
    #             pos_diff_y = coords_a2 - y_b
    #             pos_diff_z = coords_a3 - z_b
    #             distance2 = pos_diff_x * pos_diff_x + pos_diff_y * pos_diff_y + pos_diff_z * pos_diff_z
    #             mask = distance2 <= search_radius2

    #             sum(mask) == 0 && continue

    #             @inbounds for neighbor_ in 1:vectorwidth
    #                 neighbor = neighbors_block[neighbor_]
    #                 neighbor_lists.backend[length_ + 1, point] = neighbor
    #                 length_ = length_ + mask[neighbor_]
    #             end
    #         end
    #     end

    #     @inbounds neighbor_lists.lengths[point] = length_
    # end

    if sort_neighbor_lists
        sorteach!(neighbor_lists)
    end
end

@kernel cpu=false function mykernel(x, y, neighborhood_search, neighbor_lists, search_radius2)
    point = @index(Global)
    threadidx = @index(Local)

    point_coords = @inbounds extract_svector(x, Val(ndims(neighborhood_search)), point)
    cell = cell_coords(point_coords, neighborhood_search)

    length_ = @inbounds neighbor_lists.lengths[point]
    local_neighbors = @localmem Int32 (64, 128) # (groupsize, max_neighbors)

    for neighbor_cell_ in neighboring_cells(cell, neighborhood_search)
        neighbor_cell = Tuple(neighbor_cell_)
        neighbors = points_in_cell(neighbor_cell, neighborhood_search)

        for neighbor_ in eachindex(neighbors)
            neighbor = @inbounds neighbors[neighbor_]

            neighbor_point_coords = @inbounds extract_svector(y, Val(ndims(neighborhood_search)),
                                                              neighbor)

            pos_diff = point_coords - neighbor_point_coords
            distance2 = dot(pos_diff, pos_diff)

            if distance2 <= search_radius2
                length_ = length_ + 1
                @inbounds local_neighbors[threadidx, length_] = neighbor
            end
        end
    end

    @inbounds neighbor_lists.lengths[point] = length_
    for i in axes(local_neighbors, 2)
        @inbounds neighbor_lists.backend[i, point] = local_neighbors[threadidx, i]
    end
end

# Note that calling this function with `@inbounds` is not safe.
# See the comments in `foreach_neighbor_unsafe`.
@propagate_inbounds function foreach_neighbor_inner(f, neighbor_coords,
                                                    neighborhood_search::PrecomputedNeighborhoodSearch,
                                                    point, point_coords, search_radius, data)
    (; periodic_box, neighbor_lists) = neighborhood_search

    # Making the following `@inbounds` is not safe because the neighbor list
    # might not contain `point` if the NHS was not initialized correctly.
    neighbors = neighbor_lists[point]
    @fastmath @loopinfo vectorwidth=8 predicate for neighbor_ in eachindex(neighbors)
        neighbor = @inbounds neighbors[neighbor_]

        # Making this `@inbounds` is not safe because
        # `neighbor` (extracted from the neighbor list) is only guaranteed to be in bounds
        # if the neighbor lists were constructed correctly and have not been corrupted.
        neighbor_point_coords = extract_svector(neighbor_coords,
                                                Val(ndims(neighborhood_search)), neighbor)

        pos_diff = convert.(eltype(neighborhood_search),
                            point_coords - neighbor_point_coords)
        distance2 = dot(pos_diff, pos_diff)

        (pos_diff,
         distance2) = compute_periodic_distance(pos_diff, distance2, search_radius,
                                                periodic_box)

        distance = sqrt(distance2)

        # Inline to avoid loss of performance
        # compared to not using `foreach_point_neighbor`.
        data = @inline f(point, neighbor, pos_diff, distance, data)
    end

    return data
end

@propagate_inbounds function foreach_neighbor_inner(f, neighbor_coords,
                                                    neighborhood_search::PrecomputedNeighborhoodSearch,
                                                    point, point_coords, search_radius)
    (; periodic_box, neighbor_lists) = neighborhood_search

    # Making the following `@inbounds` is not safe because the neighbor list
    # might not contain `point` if the NHS was not initialized correctly.
    neighbors = neighbor_lists[point]
    for neighbor_ in eachindex(neighbors)
        neighbor = @inbounds neighbors[neighbor_]

        # Making this `@inbounds` is not safe because
        # `neighbor` (extracted from the neighbor list) is only guaranteed to be in bounds
        # if the neighbor lists were constructed correctly and have not been corrupted.
        neighbor_point_coords = extract_svector(neighbor_coords,
                                                Val(ndims(neighborhood_search)), neighbor)

        pos_diff = convert.(eltype(neighborhood_search),
                            point_coords - neighbor_point_coords)
        distance2 = dot(pos_diff, pos_diff)

        (pos_diff,
         distance2) = compute_periodic_distance(pos_diff, distance2, search_radius,
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
                                                          search_radius * (1 + nhs.update_neighborhood_search_padding),
                                                          n_points;
                                                          eachpoint)

    # For `Vector{Vector}` backend use `max_neighbors(NDIMS)` as fallback.
    # This should never be used because this backend doesn't require a `max_neighbors`.
    max_neighbors_ = max_inner_length(nhs.neighbor_lists, max_neighbors(ndims(nhs)))
    transpose_backend = transposed_backend(nhs.neighbor_lists)
    return PrecomputedNeighborhoodSearch{ndims(nhs)}(; search_radius, n_points,
                                                     periodic_box = nhs.periodic_box,
                                                     update_neighborhood_search,
                                                     backend = typeof(nhs.neighbor_lists),
                                                     transpose_backend,
                                                     max_neighbors = max_neighbors_,
                                                     sort_neighbor_lists = nhs.sort_neighbor_lists,
                                                     update_neighborhood_search_padding = nhs.update_neighborhood_search_padding)
end

@inline function freeze_neighborhood_search(search::PrecomputedNeighborhoodSearch)
    # Indicate that the neighborhood search is static and will not be updated anymore.
    # For the `PrecomputedNeighborhoodSearch`, strip the inner neighborhood search,
    # which is used only for initialization and updating.
    return PrecomputedNeighborhoodSearch{ndims(search)}(search.neighbor_lists,
                                                        search.search_radius,
                                                        search.periodic_box,
                                                        nothing,
                                                        search.sort_neighbor_lists,
                                                        search.update_neighborhood_search_padding)
end
