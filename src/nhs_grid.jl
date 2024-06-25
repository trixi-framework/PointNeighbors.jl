@doc raw"""
    GridNeighborhoodSearch{NDIMS}(; search_radius = 0.0, n_points = 0,
                                  periodic_box = nothing, threaded_update = true)

Simple grid-based neighborhood search with uniform search radius.
The domain is divided into a regular grid.
For each (non-empty) grid cell, a list of points in this cell is stored.
Instead of representing a finite domain by an array of cells, a potentially infinite domain
is represented by storing cell lists in a hash table (using Julia's `Dict` data structure),
indexed by the cell index tuple
```math
\left( \left\lfloor \frac{x}{d} \right\rfloor, \left\lfloor \frac{y}{d} \right\rfloor \right) \quad \text{or} \quad
\left( \left\lfloor \frac{x}{d} \right\rfloor, \left\lfloor \frac{y}{d} \right\rfloor, \left\lfloor \frac{z}{d} \right\rfloor \right),
```
where ``x, y, z`` are the space coordinates and ``d`` is the search radius.

To find points within the search radius around a position, only points in the neighboring
cells are considered.

See also (Chalela et al., 2021), (Ihmsen et al. 2011, Section 4.4).

As opposed to (Ihmsen et al. 2011), we do not sort the points in any way,
since not sorting makes our implementation a lot faster (although less parallelizable).

# Arguments
- `NDIMS`: Number of dimensions.

# Keywords
- `search_radius = 0.0`:    The fixed search radius. The default of `0.0` is useful together
                            with [`copy_neighborhood_search`](@ref).
- `n_points = 0`:           Total number of points. The default of `0` is useful together
                            with [`copy_neighborhood_search`](@ref).
- `periodic_box = nothing`: In order to use a (rectangular) periodic domain, pass a
                            [`PeriodicBox`](@ref).
- `threaded_update = true`: Can be used to deactivate thread parallelization in the
                            neighborhood search update. This can be one of the largest
                            sources of variations between simulations with different
                            thread numbers due to neighbor ordering changes.

## References
- M. Chalela, E. Sillero, L. Pereyra, M.A. Garcia, J.B. Cabral, M. Lares, M. Merchán.
  "GriSPy: A Python package for fixed-radius nearest neighbors search".
  In: Astronomy and Computing 34 (2021).
  [doi: 10.1016/j.ascom.2020.100443](https://doi.org/10.1016/j.ascom.2020.100443)
- Markus Ihmsen, Nadir Akinci, Markus Becker, Matthias Teschner.
  "A Parallel SPH Implementation on Multi-Core CPUs".
  In: Computer Graphics Forum 30.1 (2011), pages 99–112.
  [doi: 10.1111/J.1467-8659.2010.01832.X](https://doi.org/10.1111/J.1467-8659.2010.01832.X)
"""
struct GridNeighborhoodSearch{NDIMS, ELTYPE, CL, CB, PB} <: AbstractNeighborhoodSearch
    cell_list           :: CL
    search_radius       :: ELTYPE
    periodic_box        :: PB
    n_cells             :: NTuple{NDIMS, Int}    # Required to calculate periodic cell index
    cell_size           :: NTuple{NDIMS, ELTYPE} # Required to calculate cell index
    cell_buffer         :: CB                    # Multithreaded buffer for `update!`
    cell_buffer_indices :: Vector{Int} # Store which entries of `cell_buffer` are initialized
    threaded_update     :: Bool

    function GridNeighborhoodSearch{NDIMS}(; search_radius = 0.0, n_points = 0,
                                           periodic_box = nothing,
                                           cell_list = DictionaryCellList{NDIMS}(),
                                           threaded_update = true) where {NDIMS}
        ELTYPE = typeof(search_radius)

        cell_buffer = Array{index_type(cell_list), 2}(undef, n_points, Threads.nthreads())
        cell_buffer_indices = zeros(Int, Threads.nthreads())

        if search_radius < eps() || isnothing(periodic_box)
            # No periodicity
            n_cells = ntuple(_ -> -1, Val(NDIMS))
            cell_size = ntuple(_ -> search_radius, Val(NDIMS))
        else
            # Round up search radius so that the grid fits exactly into the domain without
            # splitting any cells. This might impact performance slightly, since larger
            # cells mean that more potential neighbors are considered than necessary.
            # Allow small tolerance to avoid inefficient larger cells due to machine
            # rounding errors.
            n_cells = Tuple(floor.(Int, (periodic_box.size .+ 10eps()) / search_radius))
            cell_size = Tuple(periodic_box.size ./ n_cells)

            if any(i -> i < 3, n_cells)
                throw(ArgumentError("the `GridNeighborhoodSearch` needs at least 3 cells " *
                                    "in each dimension when used with periodicity. " *
                                    "Please use no NHS for very small problems."))
            end
        end

        new{NDIMS, ELTYPE, typeof(cell_list), typeof(cell_buffer),
            typeof(periodic_box)}(cell_list, search_radius, periodic_box, n_cells,
                                  cell_size, cell_buffer, cell_buffer_indices,
                                  threaded_update)
    end
end

@inline Base.ndims(::GridNeighborhoodSearch{NDIMS}) where {NDIMS} = NDIMS

@inline function npoints(neighborhood_search::GridNeighborhoodSearch)
    return size(neighborhood_search.cell_buffer, 1)
end

function initialize!(neighborhood_search::GridNeighborhoodSearch,
                     x::AbstractMatrix, y::AbstractMatrix)
    initialize_grid!(neighborhood_search, y)
end

function initialize_grid!(neighborhood_search::GridNeighborhoodSearch{NDIMS},
                          y::AbstractMatrix) where {NDIMS}
    initialize_grid!(neighborhood_search, i -> extract_svector(y, Val(NDIMS), i))
end

function initialize_grid!(neighborhood_search::GridNeighborhoodSearch, coords_fun)
    (; cell_list) = neighborhood_search

    empty!(cell_list)

    for point in 1:npoints(neighborhood_search)
        # Get cell index of the point's cell
        cell = cell_coords(coords_fun(point), neighborhood_search)

        # Add point to corresponding cell
        push_cell!(cell_list, cell, point)
    end

    return neighborhood_search
end

function update!(neighborhood_search::GridNeighborhoodSearch,
                 x::AbstractMatrix, y::AbstractMatrix;
                 points_moving = (true, true))
    # The coordinates of the first set of points are irrelevant for this NHS.
    # Only update when the second set is moving.
    points_moving[2] || return neighborhood_search

    update_grid!(neighborhood_search, y)
end

# Update only with neighbor coordinates
function update_grid!(neighborhood_search::GridNeighborhoodSearch{NDIMS},
                      y::AbstractMatrix) where {NDIMS}
    update_grid!(neighborhood_search, i -> extract_svector(y, Val(NDIMS), i))
end

# Modify the existing hash table by moving points into their new cells
function update_grid!(neighborhood_search::GridNeighborhoodSearch, coords_fun)
    (; cell_list, cell_buffer, cell_buffer_indices, threaded_update) = neighborhood_search

    # Reset `cell_buffer` by moving all pointers to the beginning
    cell_buffer_indices .= 0

    # Find all cells containing points that now belong to another cell
    mark_changed_cell!(neighborhood_search, cell_list, coords_fun,
                       Val(threaded_update))

    # Iterate over all marked cells and move the points into their new cells.
    for thread in 1:Threads.nthreads()
        # Only the entries `1:cell_buffer_indices[thread]` are initialized for `thread`.
        for i in 1:cell_buffer_indices[thread]
            cell_index = cell_buffer[i, thread]
            points = cell_list[cell_index]

            # Find all points whose coordinates do not match this cell
            moved_point_indices = (i for i in eachindex(points)
                                   if !is_correct_cell(cell_list,
                                                       cell_coords(coords_fun(points[i]),
                                                                   neighborhood_search),
                                                       cell_index))

            # Add moved points to new cell
            for i in moved_point_indices
                point = points[i]
                new_cell_coords = cell_coords(coords_fun(point), neighborhood_search)

                # Add point to corresponding cell or create cell if it does not exist
                push_cell!(cell_list, new_cell_coords, point)
            end

            # Remove moved points from this cell
            deleteat_cell!(cell_list, cell_index, moved_point_indices)
        end
    end

    return neighborhood_search
end

@inline function mark_changed_cell!(neighborhood_search, cell_list, coords_fun,
                                    threaded_update::Val{true})
    # `collect` the keyset to be able to loop over it with `@threaded`
    @threaded for cell in collect(each_cell_index(cell_list))
        mark_changed_cell!(neighborhood_search, cell, coords_fun)
    end
end

@inline function mark_changed_cell!(neighborhood_search, cell_list, coords_fun,
                                    threaded_update::Val{false})
    for cell in each_cell_index(cell_list)
        mark_changed_cell!(neighborhood_search, cell, coords_fun)
    end
end

# Use this function barrier and unpack inside to avoid passing closures to Polyester.jl
# with `@batch` (`@threaded`).
# Otherwise, `@threaded` does not work here with Julia ARM on macOS.
# See https://github.com/JuliaSIMD/Polyester.jl/issues/88.
@inline function mark_changed_cell!(neighborhood_search, cell_index, coords_fun)
    (; cell_list, cell_buffer, cell_buffer_indices) = neighborhood_search

    for cell in cell_list[cell_index]
        cell = cell_coords(coords_fun(cell), neighborhood_search)

        # `cell` is a tuple, `cell_index` is the linear index used internally be the
        # cell list to store cells inside `cell`.
        # These can be identical (see `DictionaryCellList`)
        if !is_correct_cell(cell_list, cell, cell_index)
            # Mark this cell and continue with the next one.
            #
            # `cell_buffer` is preallocated,
            # but only the entries 1:i are used for this thread.
            i = cell_buffer_indices[Threads.threadid()] += 1
            cell_buffer[i, Threads.threadid()] = cell_index
            break
        end
    end
end

# 1D
@inline function eachneighbor(coords, neighborhood_search::GridNeighborhoodSearch{1})
    cell = cell_coords(coords, neighborhood_search)
    x = cell[1]
    # Generator of all neighboring cells to consider
    neighboring_cells = ((x + i) for i in -1:1)

    # Merge all lists of points in the neighboring cells into one iterator
    Iterators.flatten(points_in_cell(cell, neighborhood_search)
                      for cell in neighboring_cells)
end

# 2D
@inline function eachneighbor(coords, neighborhood_search::GridNeighborhoodSearch{2})
    cell = cell_coords(coords, neighborhood_search)
    x, y = cell
    # Generator of all neighboring cells to consider
    neighboring_cells = ((x + i, y + j) for i in -1:1, j in -1:1)

    # Merge all lists of points in the neighboring cells into one iterator
    Iterators.flatten(points_in_cell(cell, neighborhood_search)
                      for cell in neighboring_cells)
end

# 3D
@inline function eachneighbor(coords, neighborhood_search::GridNeighborhoodSearch{3})
    cell = cell_coords(coords, neighborhood_search)
    x, y, z = cell
    # Generator of all neighboring cells to consider
    neighboring_cells = ((x + i, y + j, z + k) for i in -1:1, j in -1:1, k in -1:1)

    # Merge all lists of points in the neighboring cells into one iterator
    Iterators.flatten(points_in_cell(cell, neighborhood_search)
                      for cell in neighboring_cells)
end

@inline function points_in_cell(cell_index, neighborhood_search)
    (; cell_list) = neighborhood_search

    return cell_list[periodic_cell_index(cell_index, neighborhood_search)]
end

@inline function periodic_cell_index(cell_index, neighborhood_search)
    (; n_cells, periodic_box) = neighborhood_search

    periodic_cell_index(cell_index, periodic_box, n_cells)
end

@inline periodic_cell_index(cell_index, ::Nothing, n_cells) = cell_index

@inline function periodic_cell_index(cell_index, periodic_box, n_cells)
    return rem.(cell_index, n_cells, RoundDown)
end

@inline function cell_coords(coords, neighborhood_search)
    (; periodic_box, cell_size) = neighborhood_search

    return cell_coords(coords, periodic_box, cell_size)
end

@inline function cell_coords(coords, periodic_box::Nothing, cell_size)
    return Tuple(floor_to_int.(coords ./ cell_size))
end

@inline function cell_coords(coords, periodic_box, cell_size)
    # Subtract `min_corner` to offset coordinates so that the min corner of the periodic
    # box corresponds to the (0, 0) cell of the NHS.
    # This way, there are no partial cells in the domain if the domain size is an integer
    # multiple of the cell size (which is required, see the constructor).
    offset_coords = periodic_coords(coords, periodic_box) .- periodic_box.min_corner

    return Tuple(floor_to_int.(offset_coords ./ cell_size))
end

function copy_neighborhood_search(nhs::GridNeighborhoodSearch, search_radius, n_points;
                                  eachpoint = 1:n_points)
    (; periodic_box, threaded_update) = nhs

    cell_list = copy_cell_list(nhs.cell_list, search_radius, periodic_box)

    return GridNeighborhoodSearch{ndims(nhs)}(; search_radius, n_points, periodic_box,
                                              cell_list, threaded_update)
end
