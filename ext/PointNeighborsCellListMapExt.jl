module PointNeighborsCellListMapExt

using PointNeighbors
using CellListMap: CellListMap

"""
    CellListMapNeighborhoodSearch(NDIMS; search_radius = 1.0, points_equal_neighbors = false)

Neighborhood search based on the package [CellListMap.jl](https://github.com/m3g/CellListMap.jl).
This package provides a similar implementation to the [`GridNeighborhoodSearch`](@ref)
with [`FullGridCellList`](@ref), but with better support for periodic boundaries
(periodicity is not yet hooked up here, though).
This is just a wrapper to use CellListMap.jl with the PointNeighbors.jl API.

# Arguments
- `NDIMS`: Number of dimensions.

# Keywords
- `search_radius = 1.0`:    The fixed search radius. The default of `1.0` is useful together
                            with [`copy_neighborhood_search`](@ref).
- `points_equal_neighbors = false`: Set to `true` when `x === y` in [`initialize!`](@ref) and
                                    [`update!`](@ref), i.e. when neighbors are searched for
                                    within a single set of points. This uses a CellListMap.jl
                                    `ParticleSystem` with only one set of coordinates, which
                                    only computes each pair once (with `i < j`), so the missing
                                    pairs are restored here manually by symmetry.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
    `eachindex_y` (searching within only a subset of the neighbor candidates) is not supported.
"""
struct CellListMapNeighborhoodSearch{NDIMS, PS} <: PointNeighbors.AbstractNeighborhoodSearch
    particle_system        :: PS
    points_equal_neighbors :: Bool
end

# Add dispatch on `NDIMS` to avoid method overwriting of the function in PointNeighbors.jl
function PointNeighbors.CellListMapNeighborhoodSearch(NDIMS::Integer;
                                                       search_radius = 1.0,
                                                       points_equal_neighbors = false)
    # Create a `ParticleSystem` with only one point and resize it later.
    # Non-periodic systems automatically grow their bounding box as points move,
    # so no explicit domain has to be specified here.
    x = zeros(typeof(search_radius), NDIMS, 1)

    particle_system = if points_equal_neighbors
        CellListMap.ParticleSystem(xpositions = x, unitcell = nothing,
                                   cutoff = search_radius, output = 0, parallel = true)
    else
        CellListMap.ParticleSystem(xpositions = x, ypositions = copy(x), unitcell = nothing,
                                   cutoff = search_radius, output = 0, parallel = true)
    end

    return CellListMapNeighborhoodSearch{NDIMS,
                                         typeof(particle_system)}(particle_system,
                                                                  points_equal_neighbors)
end

function PointNeighbors.search_radius(neighborhood_search::CellListMapNeighborhoodSearch)
    return neighborhood_search.particle_system.cutoff
end

@inline Base.ndims(::CellListMapNeighborhoodSearch{NDIMS}) where {NDIMS} = NDIMS

# The bounding box of the (non-periodic) `ParticleSystem` can grow whenever `x` or `y` move,
# so we conservatively require an update whenever either coordinate array changes.
@inline PointNeighbors.requires_update(::CellListMapNeighborhoodSearch) = (true, true)

function PointNeighbors.initialize!(neighborhood_search::CellListMapNeighborhoodSearch,
                                    x::AbstractMatrix, y::AbstractMatrix; kwargs...)
    PointNeighbors.update!(neighborhood_search, x, y; kwargs...)
end

function PointNeighbors.update!(neighborhood_search::CellListMapNeighborhoodSearch,
                                x::AbstractMatrix, y::AbstractMatrix;
                                points_moving = (true, true),
                                parallelization_backend = nothing, eachindex_y = nothing)
    (; particle_system, points_equal_neighbors) = neighborhood_search

    if points_equal_neighbors
        @assert x===y "when `points_equal_neighbors == true`, `x` must be equal to `y`"
        CellListMap.update!(particle_system; xpositions = x)
    else
        CellListMap.update!(particle_system; xpositions = x, ypositions = y)
    end

    return neighborhood_search
end

# The type annotation is to make Julia specialize on the type of the function.
# Otherwise, unspecialized code will cause a lot of allocations
# and heavily impact performance.
# See https://docs.julialang.org/en/v1/manual/performance-tips/#Be-aware-of-when-Julia-avoids-specializing
function PointNeighbors.foreach_point_neighbor(f::T, system_coords, neighbor_coords,
                                               neighborhood_search::CellListMapNeighborhoodSearch;
                                               parallelization_backend::PointNeighbors.ParallelizationBackend = PointNeighbors.default_backend(system_coords),
                                               points = axes(system_coords, 2)) where {T}
    (; particle_system, points_equal_neighbors) = neighborhood_search

    parallel = !(parallelization_backend isa PointNeighbors.SerialBackend)
    CellListMap.update!(particle_system; parallel)

    if points_equal_neighbors
        # With a single set of coordinates, CellListMap.jl only returns each pair once
        # (with `i < j`), so we have to use symmetry to add the missing pairs manually.
        # `0` is the returned output, which we don't use.
        CellListMap.pairwise!(particle_system) do pair, output
            pos_diff = pair.x - pair.y

            if pair.i in points
                @inline f(pair.i, pair.j, pos_diff, pair.d)
            end
            if pair.j in points
                @inline f(pair.j, pair.i, -pos_diff, pair.d)
            end

            return output
        end

        # With a single set of coordinates, only pairs with `i < j` are considered above.
        # `i == j` (self-pairs) have to be added separately.
        PointNeighbors.@threaded parallelization_backend for point in points
            zero_pos_diff = zero(PointNeighbors.SVector{ndims(neighborhood_search),
                                                        eltype(system_coords)})
            @inline f(point, point, zero_pos_diff, zero(eltype(system_coords)))
        end
    else
        CellListMap.pairwise!(particle_system) do pair, output
            if pair.i in points
                @inline f(pair.i, pair.j, pair.x - pair.y, pair.d)
            end

            return output
        end
    end

    return nothing
end

# No explicit bounds checks are performed above, so the safe and unsafe versions coincide.
function PointNeighbors.foreach_point_neighbor_unsafe(f::T, system_coords, neighbor_coords,
                                                       neighborhood_search::CellListMapNeighborhoodSearch;
                                                       parallelization_backend::PointNeighbors.ParallelizationBackend = PointNeighbors.default_backend(system_coords),
                                                       points = axes(system_coords, 2)) where {T}
    PointNeighbors.foreach_point_neighbor(f, system_coords, neighbor_coords,
                                          neighborhood_search; parallelization_backend, points)
end

function PointNeighbors.copy_neighborhood_search(nhs::CellListMapNeighborhoodSearch,
                                                 search_radius, n_points; eachpoint = 1:n_points)
    return PointNeighbors.CellListMapNeighborhoodSearch(ndims(nhs); search_radius,
                                                        points_equal_neighbors = nhs.points_equal_neighbors)
end

end
