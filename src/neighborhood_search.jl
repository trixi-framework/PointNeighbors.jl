abstract type AbstractNeighborhoodSearch end

"""
    initialize!(search::AbstractNeighborhoodSearch, x, y)

Initialize a neighborhood search with the two coordinate arrays `x` and `y`.

In general, the purpose of a neighborhood search is to find for one point in `x`
all points in `y` whose distances to that point are smaller than the search radius.
`x` and `y` are expected to be matrices, where the `i`-th column contains the coordinates
of point `i`. Note that `x` and `y` can be identical.

See also [`update!`](@ref).
"""
@inline initialize!(search::AbstractNeighborhoodSearch, x, y) = search

"""
    update!(search::AbstractNeighborhoodSearch, x, y; points_moving = (true, true))

Update an already initialized neighborhood search with the two coordinate arrays `x` and `y`.

Like [`initialize!`](@ref), but reusing the existing data structures of the already
initialized neighborhood search.
When the points only moved a small distance since the last `update!` or `initialize!`,
this is significantly faster than `initialize!`.

Not all implementations support incremental updates.
If incremental updates are not possible for an implementation, `update!` will fall back
to a regular `initialize!`.

Some neighborhood searches might not need to update when only `x` changed since the last
`update!` or `initialize!` and `y` did not change. Pass `points_moving = (true, false)`
in this case to avoid unnecessary updates.
The first flag in `points_moving` indicates if points in `x` are moving.
The second flag indicates if points in `y` are moving.

See also [`initialize!`](@ref).
"""
@inline function update!(search::AbstractNeighborhoodSearch, x, y;
                         points_moving = (true, true))
    return search
end

struct PeriodicBox{NDIMS, ELTYPE}
    min_corner :: SVector{NDIMS, ELTYPE}
    max_corner :: SVector{NDIMS, ELTYPE}
    size       :: SVector{NDIMS, ELTYPE}

    function PeriodicBox(min_corner, max_corner)
        new{length(min_corner), eltype(min_corner)}(min_corner, max_corner,
                                                    max_corner - min_corner)
    end
end

# The type annotation is to make Julia specialize on the type of the function.
# Otherwise, unspecialized code will cause a lot of allocations
# and heavily impact performance.
# See https://docs.julialang.org/en/v1/manual/performance-tips/#Be-aware-of-when-Julia-avoids-specializing
function foreach_point_neighbor(f::T, system_coords, neighbor_coords, neighborhood_search;
                                ppointes = axes(system_coords, 2),
                                parallel = true) where {T}
    foreach_point_neighbor(f, system_coords, neighbor_coords, neighborhood_search, points,
                           Val(parallel))
end

@inline function foreach_point_neighbor(f, system_coords, neighbor_coords,
                                        neighborhood_search, points, parallel::Val{true})
    @threaded for point in points
        foreach_neighbor(f, system_coords, neighbor_coords, neighborhood_search, point)
    end

    return nothing
end

@inline function foreach_point_neighbor(f, system_coords, neighbor_coords,
                                        neighborhood_search, points, parallel::Val{false})
    for point in points
        foreach_neighbor(f, system_coords, neighbor_coords, neighborhood_search, point)
    end

    return nothing
end

@inline function foreach_neighbor(f, system_coords, neighbor_system_coords,
                                  neighborhood_search, point;
                                  search_radius = neighborhood_search.search_radius)
    (; periodic_box) = neighborhood_search

    point_coords = extract_svector(system_coords, Val(ndims(neighborhood_search)), point)
    for neighbor in eachneighbor(point_coords, neighborhood_search)
        neighbor_coords = extract_svector(neighbor_system_coords,
                                          Val(ndims(neighborhood_search)), neighbor)

        pos_diff = point_coords - neighbor_coords
        distance2 = dot(pos_diff, pos_diff)

        pos_diff, distance2 = compute_periodic_distance(pos_diff, distance2, search_radius,
                                                        periodic_box)

        if distance2 <= search_radius^2
            distance = sqrt(distance2)

            # Inline to avoid loss of performance
            # compared to not using `foreach_point_neighbor`.
            @inline f(point, neighbor, pos_diff, distance)
        end
    end
end

@inline function compute_periodic_distance(pos_diff, distance2, search_radius,
                                           periodic_box::Nothing)
    return pos_diff, distance2
end

@inline function compute_periodic_distance(pos_diff, distance2, search_radius, periodic_box)
    if distance2 > search_radius^2
        # Use periodic `pos_diff`
        pos_diff -= periodic_box.size .* round.(pos_diff ./ periodic_box.size)
        distance2 = dot(pos_diff, pos_diff)
    end

    return pos_diff, distance2
end

# TODO export?
@inline function periodic_coords(coords, periodic_box)
    (; min_corner, size) = periodic_box

    # Move coordinates into the periodic box
    box_offset = floor.((coords .- min_corner) ./ size)

    return coords - box_offset .* size
end

@inline function periodic_coords(coords, periodic_box::Nothing)
    return coords
end
