"""
!!! warning "Experimental Implementation"
    This is an experimental feature and may change in any future releases.
"""

struct TreeNeighborhoodSearch{NDIMS, US, C, ELTYPE, MC} <:
       AbstractNeighborhoodSearch
    cell_list       :: C
    search_radius   :: ELTYPE
    update_strategy :: US
    marked_cells    :: MC
end

function TreeNeighborhoodSearch{NDIMS}(; search_radius = 0.0,
                                       n_points = 0, cell_list,
                                       update_strategy = nothing) where {NDIMS}
    if ndims(cell_list) != NDIMS
        throw(ArgumentError("a $(NDIMS)D cell list is required for " *
                            "a TreeNeighborhoodSearch{$(NDIMS)}"))
    end

    if isnothing(update_strategy)
        update_strategy = first(supported_update_strategies(cell_list))()
    elseif !(typeof(update_strategy) in supported_update_strategies(cell_list))
        throw(ArgumentError("$update_strategy is not a valid update strategy for " *
                            "this cell list. Available options are " *
                            "$(supported_update_strategies(cell_list))"))
    end

    if !(cell_list isa TreeGridCellList)
        throw(ArgumentError("Current implementation of `TreeNeighborhoodSearch` only supports `TreeGridCellList` as type for the cell list."))
    end

    n_cells = (2^cell_list.max_level)^NDIMS
    marked_cells = falses(n_cells)

    return TreeNeighborhoodSearch{NDIMS, typeof(update_strategy), typeof(cell_list),
                                  typeof(search_radius),
                                  typeof(marked_cells)}(cell_list,
                                                        search_radius,
                                                        update_strategy,
                                                        marked_cells)
end

@inline Base.ndims(::TreeNeighborhoodSearch{NDIMS}) where {NDIMS} = NDIMS

@inline requires_update(::TreeNeighborhoodSearch) = (false, true)

function initialize!(neighborhood_search::TreeNeighborhoodSearch,
                     x::AbstractMatrix, y::AbstractMatrix;
                     parallelization_backend = default_backend(x),
                     eachindex_y = axes(y, 2),
                     iters = 3)
    initialize_tree!(neighborhood_search, y; parallelization_backend, eachindex_y, iters)
end

function initialize_tree!(neighborhood_search, y::AbstractMatrix;
                          parallelization_backend = default_backend(y),
                          eachindex_y = axes(y, 2),
                          iters = neighborhood_search.cell_list.max_level)
    (; cell_list) = neighborhood_search
    (; cell_levels, capacity_per_cell, max_level) = cell_list
    NDIMS = ndims(cell_list)

    empty!(cell_list)
    cell_levels .= -1

    for point in eachindex_y
        point_coords = @inbounds extract_svector(y, Val(NDIMS), point)
        point_morton = morton_cell_index(point_coords, cell_list, max_level)
        push_cell!(cell_list, point_morton, point)
        cell_levels[point_morton] = max_level
    end

    for i in 1:min(iters, max_level)
        mark_merge!(neighborhood_search, level = cell_list.max_level - i,
                    capacity = capacity_per_cell)
        apply_merge!(neighborhood_search, level = cell_list.max_level - i)
    end

    return neighborhood_search
end

# For each cell on the specified `level`, we check if we can merge the subcells on `level - 1`
# The lowest level, for which we can perform the cell merging, is `cell_list.max_level - 1`
function mark_merge!(neighborhood_search;
                     level = neighborhood_search.cell_list.max_level - 1,
                     capacity = 1)
    (; cell_list, marked_cells) = neighborhood_search
    (; cells, max_level) = cell_list

    @assert 0 <= level < max_level

    marked_cells .= false
    offset = level_offset(cell_list, level)

    for i in 1:offset:length(cells)
        num_particles = 0
        for j in 0:(offset - 1)
            num_particles += length(cells[i + j])
        end
        marked_cells[i] = num_particles <= capacity
    end

    return any(marked_cells)
end

function apply_merge!(neighborhood_search;
                      level = neighborhood_search.cell_list.max_level - 1)
    (; cell_list, marked_cells) = neighborhood_search
    (; cells, max_level, cell_levels) = cell_list

    @assert 0 <= level < max_level

    offset = level_offset(cell_list, level)
    for i in eachindex(marked_cells)
        !marked_cells[i] && continue

        for j in 1:(offset - 1)
            particles = cells[i + j]

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

function mark_refine!(neighborhood_search; level = 1, capacity = 1)
    (; cell_list, marked_cells) = neighborhood_search
    (; cells, cell_levels, max_level) = cell_list

    @assert 0 <= level < max_level

    offset = level_offset(cell_list, level)
    marked_cells .= false
    for i in 1:offset:length(cells)
        if cell_levels[i] == level
            marked_cells[i] = length(cells[i]) > capacity
        end
    end

    return any(marked_cells)
end

function apply_refine!(neighborhood_search, coords; level = 1)
    (; cell_list, marked_cells) = neighborhood_search
    (; cells, max_level, cell_levels) = cell_list

    @assert 0 <= level < max_level

    NDIMS = ndims(neighborhood_search)
    n_children = 2^NDIMS
    step = level_offset(cell_list, level + 1)

    for i in eachindex(marked_cells)
        !marked_cells[i] && continue

        particles = cells[i]
        subcells = range(i, step = step, length = n_children)

        # Update cell levels
        for cell in subcells
            cell_levels[cell] = level + 1
        end

        for j in reverse(eachindex(particles))
            particle = particles[j]
            particle_coords = @inbounds extract_svector(coords,
                                                        Val(NDIMS),
                                                        particle)
            particle_z = morton_cell_index(particle_coords, cell_list)
            cell = subcells[searchsortedlast(subcells, particle_z)]

            # Only delete when the current cell is not the marked cell `i`
            cell == i && continue
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

function update_tree!(neighborhood_search::TreeNeighborhoodSearch{<:Any, ParallelUpdate},
                      y::AbstractMatrix;
                      parallelization_backend = default_backend(y),
                      eachindex_y = axes(y, 2))
    initialize_tree!(neighborhood_search, y; parallelization_backend, eachindex_y)

    return neighborhood_search
end

# Maps a Morton sub-index to its coarse base index at a specific hierarchical level
@inline function base_index(morton, level, cell_list)
    offset = level_offset(cell_list, level)
    return div(morton - 1, offset) * offset + 1
end

# Validates if a coordinate falls within domain limits
@inline is_valid_cell(cartesian, max_cartesian,
                      NDIMS) = all(1 <= cartesian[i] <= max_cartesian for i in 1:NDIMS)

# Scans upward through the hierarchy to find a merged coarse leaf containing this Morton code
@inline function find_coarse_ancestor(morton, cell_level, cell_levels, cell_list)
    for level in 0:(cell_level - 1)
        ancestor = base_index(morton, level, cell_list)
        if cell_levels[ancestor] == level
            return (ancestor, level)
        end
    end
    return (0, 0) # No coarse ancestor found
end

# Look-back algorithm that checks if this is the first time our bounding box has touched this coarse cell
@inline function is_first_encounter(current_cartesian, target_ancestor, ancestor_level,
                                    search_bounding_box, level_step, max_cartesian, NDIMS,
                                    cell_list)
    for prev_cartesian in search_bounding_box
        # If we've caught up to the current coordinate, we are the first!
        prev_cartesian == current_cartesian && return true

        !is_valid_cell(prev_cartesian, max_cartesian, NDIMS) && continue

        prev_morton_base = cartesian_to_morton(SVector(Tuple(prev_cartesian)))
        prev_morton = (prev_morton_base - 1) * level_step + 1

        if base_index(prev_morton, ancestor_level, cell_list) == target_ancestor
            return false # A previous Cartesian index in our loop already processed this
        end
    end
    return true
end

@inline function foreach_neighbor_inner(f, neighbor_system_coords,
                                        neighborhood_search::TreeNeighborhoodSearch,
                                        point, point_coords, search_radius)
    (; cell_list) = neighborhood_search
    (; max_level, cell_levels) = cell_list
    NDIMS = ndims(neighborhood_search)

    # Calculate the level required for the given search radius
    cell_level = clamp(floor(Int, log2(grid_length(cell_list) / search_radius)), 0,
                       max_level)
    # Calculate the offset for this level
    level_step = level_offset(cell_list, cell_level)
    max_cartesian = 2^cell_level

    # Define grid bounding box for the search
    cell_cartesian = cartesian_cell_coords(point_coords, cell_list, cell_level)
    search_bounding_box = CartesianIndices(ntuple(i -> (cell_cartesian[i] - 1):(cell_cartesian[i] + 1),
                                                  NDIMS))

    # Calculate distance and execute `f` if `target_cell` is a neighbor
    function evaluate_cell(target_cell)
        neighbors = points_in_cell(target_cell, neighborhood_search)
        for neighbor_ in eachindex(neighbors)
            neighbor = @inbounds neighbors[neighbor_]
            neighbor_coords = extract_svector(neighbor_system_coords, Val(NDIMS), neighbor)

            pos_diff = convert.(eltype(neighborhood_search), point_coords - neighbor_coords)
            distance2 = dot(pos_diff, pos_diff)
            pos_diff,
            distance2 = compute_periodic_distance(pos_diff, distance2, search_radius,
                                                  nothing)

            if distance2 <= search_radius^2
                @inline f(point, neighbor, pos_diff, sqrt(distance2))
            end
        end
    end

    for cartesian in search_bounding_box
        !is_valid_cell(cartesian, max_cartesian, NDIMS) && continue

        morton_base = cartesian_to_morton(SVector(Tuple(cartesian)))
        morton = (morton_base - 1) * level_step + 1

        ancestor,
        ancestor_level = find_coarse_ancestor(morton, cell_level, cell_levels, cell_list)

        if ancestor > 0
            if is_first_encounter(cartesian, ancestor, ancestor_level, search_bounding_box,
                                  level_step, max_cartesian, NDIMS, cell_list)
                evaluate_cell(ancestor)
            end
        else
            for_expanded_cell(cell_list, morton, cell_level) do subcell
                evaluate_cell(subcell)
            end
        end
    end
end

# Apply a function `f` to all subcells in a cell. 
@inline function for_expanded_cell(f, cell_list, cell, level)
    (; cell_levels) = cell_list
    offset = level_offset(cell_list, level)

    if cell_levels[cell] != -1 && cell_levels[cell] <= level
        @inline f(cell) # Run logic on the single coarse leaf
    else
        subcell = cell
        while subcell <= cell + offset - 1
            subcell_level = cell_levels[subcell]

            if subcell_level != -1
                @inline f(subcell)
                subcell += level_offset(cell_list, subcell_level)
            else
                subcell += 1
            end
        end
    end
end

@propagate_inbounds function points_in_cell(cell_index,
                                            neighborhood_search::TreeNeighborhoodSearch)
    return neighborhood_search.cell_list.cells[cell_index]
end

# TODO
function copy_neighborhood_search(nhs::TreeNeighborhoodSearch, search_radius, n_points;
                                  eachpoint = 1:n_points)
    cell_list = copy_cell_list(nhs.cell_list)

    return TreeNeighborhoodSearch{ndims(nhs)}(; search_radius, n_points, cell_list,
                                              update_strategy = nhs.update_strategy)
end
