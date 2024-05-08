mutable struct CellListMapNeighborhoodSearch{CL, B}
    cell_list :: CL
    box       :: B

    function CellListMapNeighborhoodSearch{NDIMS}(search_radius) where {NDIMS}
        # Create a cell list with only one particle and resize it later
        x = zeros(NDIMS, 1)
        box = CellListMap.Box(CellListMap.limits(x, x), search_radius)
        cell_list = CellListMap.CellList(x, x, box)

        return new{typeof(cell_list), typeof(box)}(cell_list, box)
    end
end

function initialize!(neighborhood_search::CellListMapNeighborhoodSearch,
                     x::AbstractMatrix, y::AbstractMatrix)
    update!(neighborhood_search, x, y)
end

function update!(neighborhood_search::CellListMapNeighborhoodSearch,
                 x::AbstractMatrix, y::AbstractMatrix;
                 particles_moving = (true, true))
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
function for_particle_neighbor(f::T, system_coords, neighbor_coords,
                               neighborhood_search::CellListMapNeighborhoodSearch;
                               particles = axes(system_coords, 2),
                               parallel = true) where {T}
    (; cell_list, box) = neighborhood_search

    # `0` is the returned output, which we don't use
    CellListMap.map_pairwise!(0, box, cell_list,
                              parallel = parallel) do x, y, i, j, d2, output
        # Skip all indices not in `particles`
        i in particles || return output

        pos_diff = x - y
        distance = sqrt(d2)

        @inline f(i, j, pos_diff, distance)

        return output
    end

    return nothing
end
