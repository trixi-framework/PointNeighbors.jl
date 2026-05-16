struct TreeNeighborhoodSearch{NDIMS, C, ELTYPE, US} <: AbstractNeighborhoodSearch
    cell_list       :: C
    search_radius   :: ELTYPE
    particle_z      :: Vector{UInt64} # Calculate the z-index for each particle 
    particle_idxs   :: Vector{UInt64} # Contains the particle indices, will be sorted based on z-code.
    update_strategy :: US
end

function TreeNeighborhoodSearch{NDIMS}(; cell_list, search_radius = 0.0,
                                       n_points = 0) where {NDIMS}
    particle_z = Vector{UInt64}(undef, n_points)
    particle_idxs = Vector{UInt64}(undef, n_points)

    if isnothing(update_strategy)
        update_strategy = first(supported_update_strategies(cell_list))()

    elseif !(typeof(update_strategy) in supported_update_strategies(cell_list))
        throw(ArgumentError("$update_strategy is not a valid update strategy for " *
                            "this cell list. Available options are " *
                            "$(supported_update_strategies(cell_list))"))
    end

    return TreeNeighborhoodSearch{NDIMS, typeof(cell_list), typeof(search_radius),
                                  typeof(update_strategy)}(cell_list,
                                                           search_radius, particle_z,
                                                           particle_idxs, update_strategy)
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
        mark_refine!(cell_list, level = i, capacity = capacity_per_cell)
        apply_refine!(cell_list, neighborhood_search, y, level = i)
    end

    return neighborhood_search
end

# For each cell on the specified `level`, we check if we can merge the subcells on `level - 1`
# The lowest level for which we can perform the cell merging thus is `cell_list.max_level - 1`
function mark_merge!(cell_list; level = cell_list.max_level, capacity = 1)
    (; active_cells, max_level, marked_cells) = cell_list
    @assert 2 <= level <= max_level

    offset = 4^(max_level - level)
    marked_cells .= false

    for i in 1:offset:length(active_cells)
        num_particles = 0

        for j in 0:(offset - 1)
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
        for j in 1:(offset - 1)
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

function mark_refine!(cell_list; level = 1, capacity = 1)
    (; active_cells, cell_levels, max_level, marked_cells) = cell_list
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

function apply_refine!(cell_list, neighborhood_search, coords; level = 1)
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
            particle_coords = @inbounds extract_svector(coords,
                                                        Val(ndims(neighborhood_search)),
                                                        particle)
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
function update_tree!(neighborhood_search::TreeNeighborhoodSearch{<:Any, SerialUpdate},
                      y::AbstractMatrix;
                      parallelization_backend = default_backend(y),
                      eachindex_y = axes(y, 2))
    return neighborhood_search
end

function update_tree!(neighborhood_search::TreeNeighborhoodSearch{<:Any, ParallelUpdate},
                      y::AbstractMatrix;
                      parallelization_backend = default_backend(y),
                      eachindex_y = axes(y, 2))
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
    (; cell_list, search_radius) = neighborhood_search
    (; cell_sizes, max_level) = cell_list

    cell = cell_coords(point_coords, cell_list)
    NDIMS = ndims(neighborhood_search)
    max_num_cells = 2^(NDIMS * max_level)

    # Pick the first cell size thats larger or equal the search radius. 
    cell_level = min(max(searchsortedlast(cell_sizes, search_radius, lt = >=), 0),
                     max_level)
    offset = (2^NDIMS)^(max_level - cell_level)
    cell_morton = Int(((cell - 1) / offset) + 1)
    cell_cartesian = morton2cartesian(cell_morton)

    neighbors_cartesian = CartesianIndices(ntuple(i -> (cell_cartesian[i] - 1):(cell_cartesian[i] + 1),
                                                  NDIMS))

    for neighbor_cartesian in neighbors_cartesian
        cartesian_svec = SVector(Tuple(neighbor_cartesian))
        neighbor_morton_base = cartesian2morton(cartesian_svec)
        neighbor_morton = (neighbor_morton_base - 1) * offset + 1

        if 1 <= neighbor_morton <= max_num_cells
            for_expanded_cell(cell_list, neighbor_morton, cell_level) do subcell
                neighbors = points_in_cell(subcell, neighborhood_search)

                for neighbor_ in eachindex(neighbors)
                    neighbor = @inbounds neighbors[neighbor_]
                    neighbor_coords = extract_svector(neighbor_system_coords,
                                                      Val(NDIMS), neighbor)

                    pos_diff = convert.(eltype(neighborhood_search),
                                        point_coords - neighbor_coords)
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
    end
end

@inline function for_expanded_cell(f, cell_list, cell, level)
    (; cell_levels, max_level) = cell_list

    # Note: 4^ assumes a 2D Quadtree. If this is 3D, it should be 8^.
    offset = 4^(max_level - level)
    if cell_levels[cell] <= level
        @inline f(cell) # Run the logic on the single cell
    else
        subcell = cell
        while subcell <= cell + offset - 1
            subcell_level = cell_levels[subcell]
            if subcell_level != -1
                @inline f(subcell)
            end
            subcell += 4^(max_level - subcell_level)
        end
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
