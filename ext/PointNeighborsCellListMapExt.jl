module PointNeighborsCellListMapExt

using PointNeighbors
using CellListMap: CellListMap, CellList, CellListPair

"""
    CellListMapNeighborhoodSearch(NDIMS; search_radius = 1.0, points_equal_neighbors = false)

Neighborhood search based on the package [CellListMap.jl](https://github.com/m3g/CellListMap.jl).
This package provides a similar implementation to the [`GridNeighborhoodSearch`](@ref)
with [`FullGridCellList`](@ref), but with better support for periodic boundaries.
This is just a wrapper to use CellListMap.jl with the PointNeighbors.jl API.
Note that periodic boundaries are not yet supported.

# Arguments
- `NDIMS`: Number of dimensions.

# Keywords
- `search_radius = 1.0`:    The fixed search radius. The default of `1.0` is useful together
                            with [`copy_neighborhood_search`](@ref).
- `points_equal_neighbors = false`: If `true`, a `CellListMap.CellList` is used instead of
                                    a `CellListMap.CellListPair`. This requires that `x === y`
                                    in [`initialize!`](@ref) and [`update!`](@ref).
                                    This option exists only for benchmarking purposes.
                                    It makes the main loop awkward because CellListMap.jl
                                    only computes pairs with `i < j` and PointNeighbors.jl
                                    computes all pairs, so we have to manually use symmetry
                                    to add the missing pairs.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
mutable struct CellListMapNeighborhoodSearch{CL, B}
    cell_list::CL
    # Note that we need this struct to be mutable to replace the box in `update!`
    box::B

    # Add dispatch on `NDIMS` to avoid method overwriting of the function in PointNeighbors.jl
    function PointNeighbors.CellListMapNeighborhoodSearch(NDIMS::Integer;
                                                          search_radius = 1.0,
                                                          points_equal_neighbors = false)
        # Create a cell list with only one point and resize it later
        x = zeros(NDIMS, 1)
        box = CellListMap.Box(CellListMap.limits(x, x), search_radius)

        if points_equal_neighbors
            cell_list = CellListMap.CellList(x, box)
        else
            cell_list = CellListMap.CellList(x, x, box)
        end

        return new{typeof(cell_list), typeof(box)}(cell_list, box)
    end
end

function PointNeighbors.search_radius(neighborhood_search::CellListMapNeighborhoodSearch)
    return neighborhood_search.box.cutoff
end

function Base.ndims(neighborhood_search::CellListMapNeighborhoodSearch)
    return length(neighborhood_search.box.cell_size)
end

function PointNeighbors.initialize!(neighborhood_search::CellListMapNeighborhoodSearch,
                                    x::AbstractMatrix, y::AbstractMatrix)
    PointNeighbors.update!(neighborhood_search, x, y)
end

# When `x !== y`, a `CellListPair` must be used
function PointNeighbors.update!(neighborhood_search::CellListMapNeighborhoodSearch{<:CellListPair},
                                x::AbstractMatrix, y::AbstractMatrix;
                                points_moving = (true, true))
    (; cell_list) = neighborhood_search

    # Resize box
    box = CellListMap.Box(CellListMap.limits(x, y), neighborhood_search.box.cutoff)
    neighborhood_search.box = box

    # Resize and update cell list
    CellListMap.UpdateCellList!(x, y, box, cell_list)

    # Recalculate number of batches for multithreading
    CellListMap.set_number_of_batches!(cell_list)

    return neighborhood_search
end

# When `points_equal_neighbors == true`, a `CellList` is used and `x` must be equal to `y`
function PointNeighbors.update!(neighborhood_search::CellListMapNeighborhoodSearch{<:CellList},
                                x::AbstractMatrix, y::AbstractMatrix;
                                points_moving = (true, true))
    (; cell_list) = neighborhood_search

    @assert x===y "When `points_equal_neighbors == true`, `x` must be equal to `y`"

    # Resize box
    box = CellListMap.Box(CellListMap.limits(x), neighborhood_search.box.cutoff)
    neighborhood_search.box = box

    # Resize and update cell list
    CellListMap.UpdateCellList!(x, box, cell_list)

    # Recalculate number of batches for multithreading
    CellListMap.set_number_of_batches!(cell_list)

    # Due to https://github.com/m3g/CellListMap.jl/issues/106, we have to update again
    CellListMap.UpdateCellList!(x, box, cell_list)

    return neighborhood_search
end

# The type annotation is to make Julia specialize on the type of the function.
# Otherwise, unspecialized code will cause a lot of allocations
# and heavily impact performance.
# See https://docs.julialang.org/en/v1/manual/performance-tips/#Be-aware-of-when-Julia-avoids-specializing
function PointNeighbors.foreach_point_neighbor(f::T, system_coords, neighbor_coords,
                                               neighborhood_search::CellListMapNeighborhoodSearch{<:CellListPair};
                                               points = axes(system_coords, 2),
                                               parallel = true) where {T}
    (; cell_list, box) = neighborhood_search

    # `0` is the returned output, which we don't use.
    # Note that `parallel !== false` is `true` when `parallel` is a PointNeighbors backend.
    CellListMap.map_pairwise!(0, box, cell_list,
                              parallel = parallel !== false) do x, y, i, j, d2, output
        # Skip all indices not in `points`
        i in points || return output

        pos_diff = x - y
        distance = sqrt(d2)

        @inline f(i, j, pos_diff, distance)

        return output
    end

    return nothing
end

function PointNeighbors.foreach_point_neighbor(f::T, system_coords, neighbor_coords,
                                               neighborhood_search::CellListMapNeighborhoodSearch{<:CellList};
                                               points = axes(system_coords, 2),
                                               parallel = true) where {T}
    (; cell_list, box) = neighborhood_search

    # `0` is the returned output, which we don't use.
    # Note that `parallel !== false` is `true` when `parallel` is a PointNeighbors backend.
    CellListMap.map_pairwise!(0, box, cell_list,
                              parallel = parallel !== false) do x, y, i, j, d2, output
        # Skip all indices not in `points`
        i in points || return output

        pos_diff = x - y
        distance = sqrt(d2)

        # When `points_equal_neighbors == true`, a `CellList` is used.
        # With a `CellList`, we only see each pair once and have to use symmetry manually.
        @inline f(i, j, pos_diff, distance)
        @inline f(j, i, -pos_diff, distance)

        return output
    end

    # With a `CellList`, only pairs with `i < j` are considered.
    # We can cover `i > j` with symmetry above, but `i = j` has to be computed separately.
    PointNeighbors.@threaded system_coords for point in points
        zero_pos_diff = zero(PointNeighbors.SVector{ndims(neighborhood_search),
                                                    eltype(system_coords)})
        @inline f(point, point, zero_pos_diff, zero(eltype(system_coords)))
    end

    return nothing
end

function PointNeighbors.copy_neighborhood_search(nhs::CellListMapNeighborhoodSearch{<:CellListPair},
                                                 search_radius, n_points;
                                                 eachpoint = 1:n_points)
    return PointNeighbors.CellListMapNeighborhoodSearch(ndims(nhs); search_radius,
                                                        points_equal_neighbors = false)
end

function PointNeighbors.copy_neighborhood_search(nhs::CellListMapNeighborhoodSearch{<:CellList},
                                                 search_radius, n_points;
                                                 eachpoint = 1:n_points)
    return PointNeighbors.CellListMapNeighborhoodSearch(ndims(nhs); search_radius,
                                                        points_equal_neighbors = true)
end

end
