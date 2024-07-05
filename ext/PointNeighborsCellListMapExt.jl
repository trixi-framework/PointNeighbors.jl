module PointNeighborsCellListMapExt

using PointNeighbors
using CellListMap: CellListMap

mutable struct CellListMapNeighborhoodSearch{CL, B}
    cell_list :: CL
    box       :: B

    function PointNeighbors.CellListMapNeighborhoodSearch(NDIMS, search_radius)
        # Create a cell list with only one point and resize it later
        x = zeros(NDIMS, 1)
        box = CellListMap.Box(CellListMap.limits(x, x), search_radius)
        cell_list = CellListMap.CellList(x, x, box)

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

function PointNeighbors.update!(neighborhood_search::CellListMapNeighborhoodSearch,
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

# The type annotation is to make Julia specialize on the type of the function.
# Otherwise, unspecialized code will cause a lot of allocations
# and heavily impact performance.
# See https://docs.julialang.org/en/v1/manual/performance-tips/#Be-aware-of-when-Julia-avoids-specializing
function PointNeighbors.foreach_point_neighbor(f::T, system_coords, neighbor_coords,
                                               neighborhood_search::CellListMapNeighborhoodSearch;
                                               points = axes(system_coords, 2),
                                               parallel = true) where {T}
    (; cell_list, box) = neighborhood_search

    # `0` is the returned output, which we don't use
    CellListMap.map_pairwise!(0, box, cell_list,
                              parallel = parallel) do x, y, i, j, d2, output
        # Skip all indices not in `points`
        i in points || return output

        pos_diff = x - y
        distance = sqrt(d2)

        @inline f(i, j, pos_diff, distance)

        return output
    end

    return nothing
end

function PointNeighbors.copy_neighborhood_search(nhs::CellListMapNeighborhoodSearch,
                                                 search_radius, n_points;
                                                 eachpoint = 1:n_points)
    return PointNeighbors.CellListMapNeighborhoodSearch(ndims(nhs), search_radius)
end

end
