struct TreeNeighborhoodSearch{NDIMS, C, ELTYPE} <: AbstractNeighborhoodSearch
    cell_list::C
    search_radius::ELTYPE
    particle_z       :: Vector{UInt64} # Calculate the z-index for each particle 
    particle_idxs    :: Vector{UInt64} # Contains the particle indices, will be sorted based on z-code. 
end

function TreeNeighborhoodSearch{NDIMS}(; cell_list, search_radius = 0.0, n_points = 0) where {NDIMS}
    particle_z = Vector{UInt64}(undef, n_points)
    particle_idxs = Vector{UInt64}(undef, n_points)

    return TreeNeighborhoodSearch{NDIMS, typeof(cell_list), typeof(search_radius)}(cell_list,
                                                                                   search_radius, particle_z, particle_idxs)
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
    (; cell_list, particle_z, particle_idxs) = neighborhood_search
    (; cell_levels, max_level, capacity_per_cell) = cell_list

    cell_levels .= max_level

    for point in eachindex_y
        point_coords = @inbounds extract_svector(y, Val(ndims(neighborhood_search)), point)
        point_z = morton_cell_coords(point_coords, cell_list)
        push_cell!(cell_list, point_z, point)
    end

    for i in 0:2
        mark_merge!(cell_list, level=max_level - i, capacity=capacity_per_cell)
        apply_merge!(cell_list, level=max_level -i)
    end

    return neighborhood_search
end

# For each cell on the specified `level`, we check if we can merge the subcells on `level - 1`
# The lowest level for which we can perform the cell merging thus is `cell_list.max_level - 1`
function mark_merge!(cell_list; level=cell_list.max_level, capacity=1)
    (; active_cells, max_level, marked_cells) = cell_list
    @assert 2 <= level <= max_level 

    offset = 4^(max_level - level)
    marked_cells .= false

    for i in 1:offset:length(active_cells)
        num_particles = 0

        for j in 0:offset - 1
            num_particles += length(active_cells[i + j])                
        end

        marked_cells[i] = num_particles <= capacity
    end 

    return any(marked_cells)
end

function apply_merge!(cell_list; level = cell_list.max_level - 1)
    (; active_cells, max_level, marked_cells, cell_levels) = cell_list
    @assert 2 <= level <= max_level 

    offset = 4^(max_level - level)
    for i in findall(marked_cells)
        for j in 1:offset - 1
            particles = active_cells[i + j]

            for k in reverse(eachindex(particles))
                particle = particles[k]
                deleteat_cell!(cell_list, i + j, k)
                push_cell!(cell_list, i, particle)
            end

            cell_levels[i + j] = 0 # Reset the cells
        end

        cell_levels[i] = level
    end
end 

function mark_refine!(cell_list; level=1, capacity=1)
    (; active_cells, cell_levels,  max_level, marked_cells) = cell_list
    @assert 1 <= level < max_level 

    offset = 4^(max_level - level)
    marked_cells .= false
    for i in 1:offset:length(active_cells)
        if cell_levels[i] == level 
            marked_cells[i] = length(active_cells[i]) > capacity
        end 
    end 

    return any(marked_cells)
end

function apply_refine!(cell_list, neighborhood_search, coords; level=1) 
    (; active_cells, max_level, marked_cells, cell_levels) = cell_list
    @assert 1 <= level < max_level 

    for i in findall(marked_cells)
        particles = active_cells[i]
        subcells = i .+ [0, 1, 2, 3] * 4^(max_level - level - 1)
        
        # Update the cell levels
        cell_levels[i] = 0
        for cell in subcells 
            cell_levels[cell] = level + 1
        end


        for j in reverse(eachindex(particles))
            particle = particles[j]
            particle_coords = @inbounds extract_svector(coords, Val(ndims(neighborhood_search)), particle)
            particle_z = morton_cell_coords(particle_coords, cell_list)
            cell = subcells[searchsortedlast(subcells, particle_z)]
            
            # Redundant if `cell == i`
            deleteat_cell!(cell_list, i, j)
            push_cell!(cell_list, cell, particle)
            
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

@inline function morton_cell_coords(coords, cell_list::TreeCellList,
                                    level = cell_list.max_level)
    cartesian_coords = cartesian_cell_coords(coords, cell_list, level)
    return cartesian2morton(cartesian_coords)
end

@inline function cartesian_cell_coords(coords, cell_list::TreeCellList,
                                       level = cell_list.max_level)
    (; min_corner, grid_length) = cell_list
    cell_length = grid_length / (2^level)

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
    (; min_corner, max_corner, max_level, cell_z, cell_levels) = cell_list

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
        shift_amount = 2 * (max_level - neighbor_level)

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
