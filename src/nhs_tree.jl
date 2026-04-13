struct TreeNeighborhoodSearch{NDIMS, C, ELTYPE} <: AbstractNeighborhoodSearch
    cell_list::C
    search_radius::ELTYPE
end

function TreeNeighborhoodSearch{NDIMS}(cell_list, search_radius = 0.0) where {NDIMS}
    return TreeNeighborhoodSearch{NDIMS, typeof(cell_list), typeof(search_radius)}(cell_list,
                                                                                   search_radius)
end

@inline Base.ndims(::TreeNeighborhoodSearch{NDIMS}) where {NDIMS} = NDIMS

@inline requires_update(::TreeNeighborhoodSearch) = (false, true)

function initialize!(neighborhood_search::TreeNeighborhoodSearch,
                     x::AbstractMatrix, y::AbstractMatrix;
                     parallelization_backend = default_backend(x),
                     eachindex_y = axes(y, 2))
    initialize_tree!(neighborhood_search, y; parallelization_backend, eachindex_y)
end

function initialize_tree!(neighborhood_search, y::AbstractMatrix;
                          parallelization_backend = default_backend(y),
                          eachindex_y = axes(y, 2))
    (; cell_list) = neighborhood_search
    (; particle_z, particle_indices, max_depth) = cell_list

    empty!(particle_z)

    for point in eachindex_y
        point_coords = @inbounds extract_svector(y, Val(ndims(neighborhood_search)), point)
        point_z = morton_cell_coords(point_coords, cell_list, max_depth)
        push!(particle_z, point_z)
    end

    resize!(particle_indices, length(particle_z))

    sortperm!(particle_indices, particle_z)
    sort!(particle_z)

    refine_tree!(cell_list, 1, max_depth)

    return neighborhood_search
end

function refine_tree!(cell_list, max_capacity, max_depth)
    (; cell_z, cell_levels, cell_ranges, particle_z) = cell_list

    empty!(cell_z)
    empty!(cell_levels)
    empty!(cell_ranges)

    N = length(particle_z)
    if N == 0
        return
    end

    # (start_idx, end_idx, level, prefix)
    stack = Tuple{Int, Int, UInt8, UInt64}[]
    sizehint!(stack, max_depth * 4)
    push!(stack, (1, N, UInt8(0), UInt64(0)))

    while !isempty(stack)
        # Pop the most recently added node
        start_idx, end_idx, level, prefix = pop!(stack)

        count = end_idx - start_idx + 1

        if count <= max_capacity || level == max_depth
            push!(cell_z, prefix)
            push!(cell_levels, level)
            push!(cell_ranges, start_idx:end_idx)
        else
            # Subdivide
            next_level = level + UInt8(1)
            shift_amount = 2 * (max_depth - next_level)

            # Calculate boundary Morton codes
            m1 = prefix | (UInt64(1) << shift_amount)
            m2 = prefix | (UInt64(2) << shift_amount)
            m3 = prefix | (UInt64(3) << shift_amount)

            # Find boundaries in the sorted array
            b1_rel = searchsortedfirst(@view(particle_z[start_idx:end_idx]), m1)
            b1 = start_idx + b1_rel - 1

            b2_rel = searchsortedfirst(@view(particle_z[b1:end_idx]), m2)
            b2 = b1 + b2_rel - 1

            b3_rel = searchsortedfirst(@view(particle_z[b2:end_idx]), m3)
            b3 = b2 + b3_rel - 1

            if b3 <= end_idx
                push!(stack, (b3, end_idx, next_level, m3))
            end

            if b2 < b3
                push!(stack, (b2, b3 - 1, next_level, m2))
            end

            if b1 < b2
                push!(stack, (b1, b2 - 1, next_level, m1))
            end

            if start_idx < b1
                push!(stack, (start_idx, b1 - 1, next_level, prefix))
            end
        end
    end
end

function update!(neighborhood_search::TreeNeighborhoodSearch,
                 x::AbstractMatrix, y::AbstractMatrix;
                 points_moving = (true, true), parallelization_backend = default_backend(x),
                 eachindex_y = axes(y, 2))
    # The coordinates of the first set of points are irrelevant for this NHS.
    # Only update when the second set is moving.
    points_moving[2] || return neighborhood_search

    update_tree!(neighborhood_search, y; eachindex_y, parallelization_backend)
end

# TODO
function update_tree!(neighborhood_search::TreeNeighborhoodSearch,
                      y::AbstractMatrix;
                      parallelization_backend = default_backend(y),
                      eachindex_y = axes(y, 2))
    (; cell_list) = neighborhood_search
    empty!(cell_list)
    initialize_tree!(neighborhood_search, y; parallelization_backend, eachindex_y)

    return neighborhood_search
end

# The bit-shifting in `refine_tree!` depends on 0-indexing
@inline function morton_cell_coords(coords, cell_list::TreeCellList,
                                    depth = cell_list.max_depth)
    (; min_corner, grid_length) = cell_list
    cell_length = grid_length / (2^depth)
    grid_coords = floor_to_int.((coords .- min_corner) ./ cell_length) .+ 1

    # Subtract 1 to make it a standard 0-based
    return cartesian2morton(grid_coords) - UInt64(1)
end

@inline function cartesian_cell_coords(coords, cell_list::TreeCellList,
                                       depth = cell_list.max_depth)
    (; min_corner, grid_length) = cell_list
    cell_length = grid_length / (2^depth)

    return floor_to_int.((coords .- min_corner) ./ cell_length) .+ 1
end

@inline function foreach_neighbor(f, neighbor_system_coords,
                                  neighborhood_search::TreeNeighborhoodSearch,
                                  point, point_coords, search_radius)
    for neighbor_cell_ in neighboring_cells(point_coords, neighborhood_search)
        neighbors = points_in_cell(neighbor_cell_, neighborhood_search)

        for neighbor_ in eachindex(neighbors)
            neighbor = @inbounds neighbors[neighbor_]
            neighbor_coords = extract_svector(neighbor_system_coords,
                                              Val(ndims(neighborhood_search)), neighbor)

            pos_diff = convert.(eltype(neighborhood_search), point_coords - neighbor_coords)
            distance2 = dot(pos_diff, pos_diff)

            pos_diff,
            distance2 = compute_periodic_distance(pos_diff, distance2,
                                                  search_radius, nothing)

            if distance2 <= search_radius^2
                distance = sqrt(distance2)
                @inline f(point, neighbor, pos_diff, distance)
            end
        end
    end
end

# Returns the indices in `cell_z` for the neighboring cells of the point at `coords`. 
@inline function neighboring_cells(coords, neighborhood_search::TreeNeighborhoodSearch)
    (; cell_list, search_radius) = neighborhood_search
    (; min_corner, max_corner, max_depth, cell_z, cell_levels) = cell_list

    min_corner_point = maximum([coords .- search_radius, min_corner])
    max_corner_point = minimum([coords .+ search_radius, max_corner])

    min_cell_point = cartesian_cell_coords(min_corner_point, cell_list)
    max_cell_point = cartesian_cell_coords(max_corner_point, cell_list)

    visited_cells = BitSet()
    neighboring_cells = BitSet()

    for i in min_cell_point[1]:max_cell_point[1], j in min_cell_point[2]:max_cell_point[2]
        candidate_z = cartesian2morton([i, j]) - UInt64(1)
        neighbor_idx = searchsortedlast(cell_z, candidate_z)
        already_visited = neighbor_idx in visited_cells
        push!(visited_cells, neighbor_idx)

        if neighbor_idx == 0 || already_visited
            continue
        end

        neighbor_prefix = cell_z[neighbor_idx]
        neighbor_level = cell_levels[neighbor_idx]
        shift_amount = 2 * (max_depth - neighbor_level)

        candidate_prefix = (candidate_z >> shift_amount) << shift_amount

        if neighbor_prefix == candidate_prefix
            push!(neighboring_cells, neighbor_idx)
        end
    end

    return neighboring_cells
end

@inline function eachneighbor(coords, neighborhood_search::TreeNeighborhoodSearch)

    # Merge all lists of points in the neighboring cells into one iterator
    Iterators.flatten(points_in_cell(cell, neighborhood_search)
                      for cell in neighboring_cells(coords, neighborhood_search))
end

# Expects a `cell_index` in the range 1:length(cell_ranges).
@propagate_inbounds function points_in_cell(cell_index, neighborhood_search)
    (; cell_list) = neighborhood_search
    (; cell_ranges, particle_indices) = cell_list

    return particle_indices[cell_ranges[cell_index]]
end

function copy_neighborhood_search(nhs::TreeNeighborhoodSearch, search_radius, n_points;
                                  eachpoint = 1:n_points)
end
