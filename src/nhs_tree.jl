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

    empty!(cell_list)
    cell_levels .= -1

    # Initialize the grid by pushing every point to the cell on the top-most level. 
    cell_levels[1] = 0
    for point in eachindex_y
        push_cell!(cell_list, 1, point)
    end

    for i in 0:2
        mark_refine!(cell_list, level=i, capacity=capacity_per_cell)
        apply_refine!(cell_list, neighborhood_search, y, level=i)
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

            cell_levels[i + j] = -1 # Reset the cells
        end

        cell_levels[i] = level
    end
end 

function mark_refine!(cell_list; level=1, capacity=1)
    (; active_cells, cell_levels,  max_level, marked_cells) = cell_list
    @assert 0 <= level < max_level 

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
    @assert 0 <= level < max_level 

    for i in findall(marked_cells)
        particles = active_cells[i]
        subcells = i .+ [0, 1, 2, 3] * 4^(max_level - level - 1)
        
        # Update cell levels
        for cell in subcells 
            cell_levels[cell] = level + 1
        end

        for j in reverse(eachindex(particles))
            particle = particles[j]
            particle_coords = @inbounds extract_svector(coords, Val(ndims(neighborhood_search)), particle)
            particle_z = morton_cell_index(particle_coords, cell_list)
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

# For a given point, compute the cell it belongs to based on its coordinates
@inline function cell_coords(coords, cell_list)
    (; max_level, cell_levels) = cell_list 

    for level in 1:max_level
        # For a given level, we identify a cell on this level with the smallest cell on the `max_level` that is part of it. 
        # For example, for `max_level = 2`, we identify cell 2 on level 1 with cell 5. 
        morton_code = morton_cell_index(coords, cell_list, level)
        offset = 4^(max_level - level) # TODO: Move this into an SVector and store as property like `cell_list.level_offsets[level]`
        cell = (morton_code - 1) * offset + 1 # Map Morton code to the cell index we identify the cell with 

        if cell_levels[cell] != -1
            return cell
        end
    end

    return 0 
end

@inline function morton_cell_index(coords, cell_list::TreeCellList,
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

    (; cell_list) = neighborhood_search
    cell = cell_coords(point_coords, cell_list)

    for neighbor_cell_ in neighboring_cells(cell, neighborhood_search)
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
@inline function neighboring_cells(cell, neighborhood_search::TreeNeighborhoodSearch)
    (; cell_list) = neighborhood_search
    (; cell_levels, max_level) = cell_list

    NDIMS = ndims(neighborhood_search)     
    level = cell_levels[cell]
    offset = 4^(max_level - level)
    morton_code = Int(((cell - 1) / offset) + 1)

    cartesian_code = morton2cartesian(morton_code)

    neighbors_cartesian = CartesianIndices(ntuple(i -> (cartesian_code[i] - 1):(cartesian_code[i] + 1), NDIMS))
    neighbors_morton = [cartesian2morton(collect(Tuple(neighbor_cartesian))) for neighbor_cartesian in neighbors_cartesian]

    # Filter out adjacent cells at the grid boundary that are not part of the grid, 
    # e.g. in 2D we filter out 5 of the 8 neighbors of cell 1.
    max_num_cells = 2^(NDIMS * max_level)
    neighbors_morton = [(neighbor_morton -1) * offset + 1 for neighbor_morton in neighbors_morton]
    neighbors_morton = neighbors_morton[1 .<= neighbors_morton .<= max_num_cells]
    neighbors_morton = 
    
    return neighbors_morton
end

function expand_cell(cell_list, cell, level)
    (; cell_levels, max_level) = cell_list
    offset = 4^(max_level - level)

    if cell_level[cell] < level
        cell_candidates = cell_levels[cell, cell + offset - 1]
        return findall(cell_candidates .!= 0)
    else 
        return [cell]
    end
end

@inline function eachneighbor(coords, neighborhood_search::TreeNeighborhoodSearch)

    # Merge all lists of points in the neighboring cells into one iterator
    Iterators.flatten(points_in_cell(cell, neighborhood_search)
                      for cell in neighboring_cells(coords, neighborhood_search))
end

@propagate_inbounds function points_in_cell(cell_index, neighborhood_search)
    return neighborhood_search.cell_list.active_cells[cell_index]
end

function copy_neighborhood_search(nhs::TreeNeighborhoodSearch, search_radius, n_points;
                                  eachpoint = 1:n_points)
end
