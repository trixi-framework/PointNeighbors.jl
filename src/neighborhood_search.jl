abstract type AbstractNeighborhoodSearch end

@inline search_radius(search::AbstractNeighborhoodSearch) = search.search_radius
@inline Base.eltype(search::AbstractNeighborhoodSearch) = eltype(search_radius(search))

"""
    requires_update(search::AbstractNeighborhoodSearch)

Returns a tuple `(x_changed, y_changed)` indicating if this type of neighborhood search
requires an update when the coordinates of the points in `x` or `y` change.
"""
function requires_update(::AbstractNeighborhoodSearch)
    error("`requires_update` not implemented for this neighborhood search.")
end

"""
    initialize!(search::AbstractNeighborhoodSearch, x, y;
                parallelization_backend = default_backend(x),
                eachindex_y = axes(y, 2))

Initialize a neighborhood search with the two coordinate arrays `x` and `y`.

In general, the purpose of a neighborhood search is to find for one point in `x`
all points in `y` whose distances to that point are smaller than the search radius.
`x` and `y` are expected to be matrices, where the `i`-th column contains the coordinates
of point `i`. Note that `x` and `y` can be identical.

If the neighborhood search type supports parallelization, the keyword argument
`parallelization_backend` can be used to specify a parallelization backend.
See [`@threaded`](@ref) for a list of available backends.

Optionally, when points in `y` are to be ignored, the keyword argument `eachindex_y` can be
passed to specify the indices of the points in `y` that are to be used.

See also [`update!`](@ref).
"""
@inline function initialize!(search::AbstractNeighborhoodSearch, x, y;
                             parallelization_backend = default_backend(x),
                             eachindex_y = axes(y, 2))
    return search
end

"""
    update!(search::AbstractNeighborhoodSearch, x, y; points_moving = (true, true),
            parallelization_backend = default_backend(x),
            eachindex_y = axes(y, 2))

Update an already initialized neighborhood search with the two coordinate arrays `x` and `y`.

Like [`initialize!`](@ref), but potentially reusing the existing data structures
of the already initialized neighborhood search.
When the points only moved a small distance since the last `update!` or `initialize!`,
this can be significantly faster than `initialize!`.

Not all implementations support incremental updates.
If incremental updates are not possible for an implementation, `update!` will fall back
to a regular `initialize!`.

Some neighborhood searches might not need to update when only `x` changed since the last
`update!` or `initialize!` and `y` did not change. Pass `points_moving = (true, false)`
in this case to avoid unnecessary updates.
The first flag in `points_moving` indicates if points in `x` are moving.
The second flag indicates if points in `y` are moving.

If the neighborhood search type supports parallelization, the keyword argument
`parallelization_backend` can be used to specify a parallelization backend.
See [`@threaded`](@ref) for a list of available backends.

Optionally, when points in `y` are to be ignored, the keyword argument `eachindex_y` can be
passed to specify the indices of the points in `y` that are to be used.

See also [`initialize!`](@ref).
"""
@inline function update!(search::AbstractNeighborhoodSearch, x, y;
                         points_moving = (true, true),
                         parallelization_backend = default_backend(x),
                         eachindex_y = axes(y, 2))
    return search
end

"""
    copy_neighborhood_search(search::AbstractNeighborhoodSearch, search_radius, n_points;
                             eachpoint = 1:n_points)

Create a new **uninitialized** neighborhood search of the same type and with the same
configuration options as `search`, but with a different search radius and number of points.

The [`TrivialNeighborhoodSearch`](@ref) also requires an iterator `eachpoint`, which most
of the time will be `1:n_points`. If the `TrivialNeighborhoodSearch` is never going to be
used, the keyword argument `eachpoint` can be ignored.

This is useful when a simulation code requires multiple neighborhood searches of the same
kind. One can then just pass an empty neighborhood search as a template and use
this function inside the simulation code to generate similar neighborhood searches with
different search radii and different numbers of points.
```jldoctest; filter = r"GridNeighborhoodSearch{2,.*"
# Template
nhs = GridNeighborhoodSearch{2}()

# Inside the simulation code, generate similar neighborhood searches
nhs1 = copy_neighborhood_search(nhs, 1.0, 100)

# output
GridNeighborhoodSearch{2, Float64, ...}(...)
```
"""
@inline function copy_neighborhood_search(search::AbstractNeighborhoodSearch,
                                          search_radius, n_points; eachpoint = 1:n_points)
    return nothing
end

@inline function freeze_neighborhood_search(search::AbstractNeighborhoodSearch)
    # Indicate that the neighborhood search is static and will not be updated anymore.
    # Some implementations might use this to strip unnecessary data structures for updating.
    # Notably, this is used for the `PrecomputedNeighborhoodSearch` to strip a potentially
    # not GPU-compatible inner neighborhood search that is used only for initialization.
    return search
end

"""
    PeriodicBox(; min_corner, max_corner)

Define a rectangular (axis-aligned) periodic domain.

# Keywords
- `min_corner`: Coordinates of the domain corner in negative coordinate directions.
- `max_corner`: Coordinates of the domain corner in positive coordinate directions.
"""
struct PeriodicBox{NDIMS, ELTYPE}
    min_corner :: SVector{NDIMS, ELTYPE}
    max_corner :: SVector{NDIMS, ELTYPE}
    size       :: SVector{NDIMS, ELTYPE}

    function PeriodicBox(; min_corner, max_corner)
        min_corner_ = SVector(Tuple(min_corner))
        max_corner_ = SVector(Tuple(max_corner))

        new{length(min_corner), eltype(min_corner)}(min_corner_, max_corner_,
                                                    max_corner_ - min_corner_)
    end
end

@inline Base.eltype(::PeriodicBox{<:Any, ELTYPE}) where {ELTYPE} = ELTYPE

"""
    foreach_point_neighbor(f, system_coords, neighbor_coords, neighborhood_search;
                           parallelization_backend = default_backend(system_coords),
                           points = axes(system_coords, 2))

Loop for each point in `system_coords` over all points in `neighbor_coords` whose distances
to that point are smaller than the search radius and execute the function `f(i, j, pos_diff, d)`,
where
- `i` is the column index of the point in `system_coords`,
- `j` the column index of the neighbor in `neighbor_coords`,
- `pos_diff` the vector ``x - y`` where ``x`` denotes the coordinates of the point
  (`system_coords[:, i]`) and ``y`` the coordinates of the neighbor (`neighbor_coords[:, j]`),
- `d` the distance between `x` and `y`.

Note that `system_coords` and `neighbor_coords` can be identical.

!!! warning
    The `neighborhood_search` must have been initialized or updated with `system_coords`
    as first coordinate array and `neighbor_coords` as second coordinate array.
    This can be skipped for certain implementations. See [`requires_update`](@ref).

# Arguments
- `f`: The function explained above.
- `system_coords`: A matrix where the `i`-th column contains the coordinates of point `i`.
- `neighbor_coords`: A matrix where the `j`-th column contains the coordinates of point `j`.
- `neighborhood_search`: A neighborhood search initialized or updated with `system_coords`
                         as first coordinate array and `neighbor_coords` as second
                         coordinate array.

# Keywords
- `points`: Loop over these point indices. By default all columns of `system_coords`.
- `parallelization_backend`: Run the outer loop over `points` in parallel with the
                             specified backend. By default, the backend is selected
                             automatically based on the type of `system_coords`.
                             See [`@threaded`](@ref) for a list of available backends.

See also [`foreach_point_neighbor_unsafe`](@ref), [`initialize!`](@ref), [`update!`](@ref).
"""
function foreach_point_neighbor(f::T, system_coords, neighbor_coords, neighborhood_search;
                                parallelization_backend::ParallelizationBackend = default_backend(system_coords),
                                points = axes(system_coords, 2)) where {T}
    # The type annotation above is to make Julia specialize on the type of the function.
    # Otherwise, unspecialized code will cause a lot of allocations
    # and heavily impact performance.
    # See https://docs.julialang.org/en/v1/manual/performance-tips/#Be-aware-of-when-Julia-avoids-specializing

    # Explicit bounds check before the hot loop (or GPU kernel)
    @boundscheck checkbounds(system_coords, ndims(neighborhood_search), points)

    @threaded parallelization_backend for point in points
        # Now we can safely assume that `point` is inbounds
        @inbounds foreach_neighbor(f, system_coords, neighbor_coords,
                                   neighborhood_search, point)
    end

    return nothing
end

"""
    foreach_point_neighbor_unsafe(f, system_coords, neighbor_coords, neighborhood_search;
                                  parallelization_backend = default_backend(system_coords),
                                  points = axes(system_coords, 2))

Like [`foreach_point_neighbor`](@ref), but skips **all** bounds checks inside the
threaded loop or GPU kernel.
See [`foreach_neighbor_unsafe`](@ref) for more details on which bounds checks are skipped.

!!! warning
    The `neighborhood_search` must have been initialized or updated with `system_coords`
    as first coordinate array and `neighbor_coords` as second coordinate array.
    This can be skipped for certain implementations. See [`requires_update`](@ref).

!!! warning
    Use this only when `neighborhood_search` is known to be initialized correctly for
    `system_coords` and `neighbor_coords`.
"""
function foreach_point_neighbor_unsafe(f::T, system_coords, neighbor_coords,
                                       neighborhood_search;
                                       parallelization_backend::ParallelizationBackend = default_backend(system_coords),
                                       points = axes(system_coords, 2)) where {T}
    # Explicit bounds check before the hot loop (or GPU kernel)
    @boundscheck checkbounds(system_coords, ndims(neighborhood_search), points)

    @threaded parallelization_backend for point in points
        foreach_neighbor_unsafe(f, system_coords, neighbor_coords,
                                neighborhood_search, point)
    end

    return nothing
end

"""
    foreach_neighbor(f, system_coords, neighbor_coords,
                     neighborhood_search::AbstractNeighborhoodSearch, point;
                     search_radius = search_radius(neighborhood_search))

Loop over all neighbors of `point` and execute `f(i, j, pos_diff, d)` for every
neighbor within `search_radius`, where `i` is `point`, `j` is the neighbor index,
`pos_diff` is the vector from neighbor to point, and `d` is the distance.

This method performs bounds checks, even when called with `@inbounds`.
`@inbounds` only skips the bounds check for loading the coordinates of `point`
from `system_coords`.
See [`foreach_neighbor_unsafe`](@ref) for a version that skips all bounds checks.
"""
@propagate_inbounds function foreach_neighbor(f, system_coords, neighbor_coords,
                                              neighborhood_search::AbstractNeighborhoodSearch,
                                              point, args...;
                                              search_radius = search_radius(neighborhood_search))
    # Due to https://github.com/JuliaLang/julia/issues/30411, we cannot just remove
    # a `@boundscheck` by calling this function with `@inbounds` because it has a kwarg.
    # We have to use `@propagate_inbounds`, which will also remove boundschecks
    # in the neighbor loop, which is not safe (that's what `foreach_neighbor_unsafe` is for).
    # To avoid this, we have to use a function barrier to disable the `@inbounds` again.
    point_coords = extract_svector(system_coords, Val(ndims(neighborhood_search)), point)

    foreach_neighbor(f, neighbor_coords, neighborhood_search,
                     point, point_coords, search_radius, args...)
end

# This is a function barrier to prevent the `@inbounds` in `foreach_neighbor`
# from propagating into the neighbor loop, which is not safe.
@inline function foreach_neighbor(f, neighbor_coords,
                                  neighborhood_search::AbstractNeighborhoodSearch,
                                  point, point_coords, search_radius, args...)
    foreach_neighbor_inner(f, neighbor_coords, neighborhood_search,
                           point, point_coords, search_radius, args...)
end

"""
    foreach_neighbor_unsafe(f, system_coords, neighbor_coords,
                            neighborhood_search::AbstractNeighborhoodSearch, point;
                            search_radius = search_radius(neighborhood_search))

Like [`foreach_neighbor`](@ref), but skips **all** bounds checks.

`foreach_neighbor` performs the following bounds checks that are skipped here:
- Check if `point` is in bounds of `system_coords`. This is the only bounds check
  that is skipped when calling `foreach_neighbor` with `@inbounds`, and the only one that
  is safe to skip when `point` is guaranteed to be in bounds of `system_coords`.
- Check if the neighbors of `point` are in bounds of `neighbor_coords`.
  This is not safe to skip when the neighborhood search was not initialized correctly;
  for example when initialized with coordinate arrays of different sizes than the ones
  passed to this function or when the internal data structures have been tampered with.
  In this case, the cell lists (for [`GridNeighborhoodSearch`](@ref)) or neighbor lists
  (for [`PrecomputedNeighborhoodSearch`](@ref)) might contain indices that are out of
  bounds for `neighbor_coords`.
- With `GridNeighborhoodSearch` and [`FullGridCellList`](@ref), check if the neighboring
  cells are in bounds of the grid. Again, this is not safe to skip when the neighborhood
  search might not have been initialized correctly.
- With `PrecomputedNeighborhoodSearch`, check if `point` is in bounds of the neighbor lists.
  Again, this is not safe to skip when the neighborhood search was not initialized correctly.

Note that all these bounds checks are safe to skip if
- `point` is guaranteed to be in bounds of `system_coords`,
- the neighborhood search was initialized correctly with `system_coords`
  and `neighbor_coords` and has not been tampered with since then.

!!! warning
    Use this only when `point` is known to be in bounds of `system_coords`
    and when `neighborhood_search` is known to be initialized correctly for
    `system_coords` and `neighbor_coords`.
"""
@inline function foreach_neighbor_unsafe(f, system_coords, neighbor_coords,
                                         neighborhood_search::AbstractNeighborhoodSearch,
                                         point, args...;
                                         search_radius = search_radius(neighborhood_search))
    point_coords = @inbounds extract_svector(system_coords, Val(ndims(neighborhood_search)),
                                             point)

    @inbounds foreach_neighbor_inner(f, neighbor_coords, neighborhood_search,
                                     point, point_coords, search_radius, args...)
end

# This is the generic function that is called for `TrivialNeighborhoodSearch`.
# For `GridNeighborhoodSearch`, a specialized function is used for slightly better
# performance. `PrecomputedNeighborhoodSearch` can skip the distance check altogether.
# Note that calling this function with `@inbounds` is not safe.
# See the comments in `foreach_neighbor_unsafe`.
@propagate_inbounds function foreach_neighbor_inner(f, neighbor_coords,
                                                    neighborhood_search::AbstractNeighborhoodSearch,
                                                    point, point_coords, search_radius)
    (; periodic_box) = neighborhood_search

    for neighbor in eachneighbor(point_coords, neighborhood_search)
        neighbor_point_coords = extract_svector(neighbor_coords,
                                                Val(ndims(neighborhood_search)), neighbor)

        pos_diff = convert.(eltype(neighborhood_search),
                            point_coords - neighbor_point_coords)
        distance2 = dot(pos_diff, pos_diff)

        (pos_diff,
         distance2) = compute_periodic_distance(pos_diff, distance2, search_radius,
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
