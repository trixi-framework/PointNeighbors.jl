@doc raw"""
    GridNeighborhoodSearch{NDIMS}(search_radius, n_particles; periodic_box_min_corner=nothing,
                                  periodic_box_max_corner=nothing, threaded_nhs_update=true)

Simple grid-based neighborhood search with uniform search radius.
The domain is divided into a regular grid.
For each (non-empty) grid cell, a list of particles in this cell is stored.
Instead of representing a finite domain by an array of cells, a potentially infinite domain
is represented by storing cell lists in a hash table (using Julia's `Dict` data structure),
indexed by the cell index tuple
```math
\left( \left\lfloor \frac{x}{d} \right\rfloor, \left\lfloor \frac{y}{d} \right\rfloor \right) \quad \text{or} \quad
\left( \left\lfloor \frac{x}{d} \right\rfloor, \left\lfloor \frac{y}{d} \right\rfloor, \left\lfloor \frac{z}{d} \right\rfloor \right),
```
where ``x, y, z`` are the space coordinates and ``d`` is the search radius.

To find particles within the search radius around a point, only particles in the neighboring
cells are considered.

See also (Chalela et al., 2021), (Ihmsen et al. 2011, Section 4.4).

As opposed to (Ihmsen et al. 2011), we do not sort the particles in any way,
since not sorting makes our implementation a lot faster (although less parallelizable).

# Arguments
- `NDIMS`:          Number of dimensions.
- `search_radius`:  The uniform search radius.
- `n_particles`:    Total number of particles.

# Keywords
- `periodic_box_min_corner`:    In order to use a (rectangular) periodic domain, pass the
                                coordinates of the domain corner in negative coordinate
                                directions.
- `periodic_box_max_corner`:    In order to use a (rectangular) periodic domain, pass the
                                coordinates of the domain corner in positive coordinate
                                directions.
- `threaded_nhs_update=true`:              Can be used to deactivate thread parallelization in the neighborhood search update.
                                This can be one of the largest sources of variations between simulations
                                with different thread numbers due to particle ordering changes.

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
struct GridNeighborhoodSearch{NDIMS, ELTYPE, CL, PB}
    cell_list           :: CL
    search_radius       :: ELTYPE
    periodic_box        :: PB
    n_cells             :: NTuple{NDIMS, Int}    # Required to calculate periodic cell index
    cell_size           :: NTuple{NDIMS, ELTYPE} # Required to calculate cell index
    cell_buffer         :: Array{NTuple{NDIMS, Int}, 2} # Multithreaded buffer for `update!`
    cell_buffer_indices :: Vector{Int} # Store which entries of `cell_buffer` are initialized
    threaded_nhs_update :: Bool

    function GridNeighborhoodSearch{NDIMS}(search_radius, n_particles;
                                           periodic_box_min_corner = nothing,
                                           periodic_box_max_corner = nothing,
                                           threaded_nhs_update = true) where {NDIMS}
        ELTYPE = typeof(search_radius)
        cell_list = DictionaryCellList{NDIMS}()

        cell_buffer = Array{NTuple{NDIMS, Int}, 2}(undef, n_particles, Threads.nthreads())
        cell_buffer_indices = zeros(Int, Threads.nthreads())

        if search_radius < eps() ||
           (periodic_box_min_corner === nothing && periodic_box_max_corner === nothing)

            # No periodicity
            periodic_box = nothing
            n_cells = ntuple(_ -> -1, Val(NDIMS))
            cell_size = ntuple(_ -> search_radius, Val(NDIMS))
        elseif periodic_box_min_corner !== nothing && periodic_box_max_corner !== nothing
            periodic_box = PeriodicBox(periodic_box_min_corner, periodic_box_max_corner)

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
        else
            throw(ArgumentError("`periodic_box_min_corner` and `periodic_box_max_corner` " *
                                "must either be both `nothing` or both an array or tuple"))
        end

        new{NDIMS, ELTYPE, typeof(cell_list),
            typeof(periodic_box)}(cell_list, search_radius, periodic_box, n_cells,
                                  cell_size, cell_buffer, cell_buffer_indices,
                                  threaded_nhs_update)
    end
end

@inline Base.ndims(neighborhood_search::GridNeighborhoodSearch{NDIMS}) where {NDIMS} = NDIMS

@inline function nparticles(neighborhood_search::GridNeighborhoodSearch)
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

    for particle in 1:nparticles(neighborhood_search)
        # Get cell index of the particle's cell
        cell = cell_coords(coords_fun(particle), neighborhood_search)

        # Add particle to corresponding cell
        push_cell!(cell_list, cell, particle)
    end

    return neighborhood_search
end

function update!(neighborhood_search::GridNeighborhoodSearch,
                 x::AbstractMatrix, y::AbstractMatrix;
                 particles_moving = (true, true))
    # The coordinates of the first set of particles are irrelevant for this NHS.
    # Only update when the second set is moving.
    particles_moving[2] || return neighborhood_search

    update_grid!(neighborhood_search, y)
end

# Update only with neighbor coordinates
function update_grid!(neighborhood_search::GridNeighborhoodSearch{NDIMS},
                      y::AbstractMatrix) where {NDIMS}
    update_grid!(neighborhood_search, i -> extract_svector(y, Val(NDIMS), i))
end

# Modify the existing hash table by moving particles into their new cells
function update_grid!(neighborhood_search::GridNeighborhoodSearch, coords_fun)
    (; cell_list, cell_buffer, cell_buffer_indices, threaded_nhs_update) = neighborhood_search

    # Reset `cell_buffer` by moving all pointers to the beginning
    cell_buffer_indices .= 0

    # Find all cells containing particles that now belong to another cell
    mark_changed_cell!(neighborhood_search, cell_list, coords_fun,
                       Val(threaded_nhs_update))

    # Iterate over all marked cells and move the particles into their new cells.
    for thread in 1:Threads.nthreads()
        # Only the entries `1:cell_buffer_indices[thread]` are initialized for `thread`.
        for i in 1:cell_buffer_indices[thread]
            cell = cell_buffer[i, thread]
            particles = cell_list[cell]

            # Find all particles whose coordinates do not match this cell
            moved_particle_indices = (i for i in eachindex(particles)
                                      if cell_coords(coords_fun(particles[i]),
                                                     neighborhood_search) != cell)

            # Add moved particles to new cell
            for i in moved_particle_indices
                particle = particles[i]
                new_cell_coords = cell_coords(coords_fun(particle), neighborhood_search)

                # Add particle to corresponding cell or create cell if it does not exist
                push_cell!(cell_list, new_cell_coords, particle)
            end

            # Remove moved particles from this cell
            deleteat_cell!(cell_list, cell, moved_particle_indices)
        end
    end

    return neighborhood_search
end

@inline function mark_changed_cell!(neighborhood_search, cell_list, coords_fun,
                                    threaded_nhs_update::Val{true})
    # `collect` the keyset to be able to loop over it with `@threaded`
    @threaded for cell in collect(eachcell(cell_list))
        mark_changed_cell!(neighborhood_search, cell, coords_fun)
    end
end

@inline function mark_changed_cell!(neighborhood_search, cell_list, coords_fun,
                                    threaded_nhs_update::Val{false})
    for cell in eachcell(cell_list)
        mark_changed_cell!(neighborhood_search, cell, coords_fun)
    end
end

# Use this function barrier and unpack inside to avoid passing closures to Polyester.jl
# with `@batch` (`@threaded`).
# Otherwise, `@threaded` does not work here with Julia ARM on macOS.
# See https://github.com/JuliaSIMD/Polyester.jl/issues/88.
@inline function mark_changed_cell!(neighborhood_search, cell, coords_fun)
    (; cell_list, cell_buffer, cell_buffer_indices) = neighborhood_search

    for particle in cell_list[cell]
        if cell_coords(coords_fun(particle), neighborhood_search) != cell
            # Mark this cell and continue with the next one.
            #
            # `cell_buffer` is preallocated,
            # but only the entries 1:i are used for this thread.
            i = cell_buffer_indices[Threads.threadid()] += 1
            cell_buffer[i, Threads.threadid()] = cell
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

    # Merge all lists of particles in the neighboring cells into one iterator
    Iterators.flatten(particles_in_cell(cell, neighborhood_search)
                      for cell in neighboring_cells)
end

# 2D
@inline function eachneighbor(coords, neighborhood_search::GridNeighborhoodSearch{2})
    cell = cell_coords(coords, neighborhood_search)
    x, y = cell
    # Generator of all neighboring cells to consider
    neighboring_cells = ((x + i, y + j) for i in -1:1, j in -1:1)

    # Merge all lists of particles in the neighboring cells into one iterator
    Iterators.flatten(particles_in_cell(cell, neighborhood_search)
                      for cell in neighboring_cells)
end

# 3D
@inline function eachneighbor(coords, neighborhood_search::GridNeighborhoodSearch{3})
    cell = cell_coords(coords, neighborhood_search)
    x, y, z = cell
    # Generator of all neighboring cells to consider
    neighboring_cells = ((x + i, y + j, z + k) for i in -1:1, j in -1:1, k in -1:1)

    # Merge all lists of particles in the neighboring cells into one iterator
    Iterators.flatten(particles_in_cell(cell, neighborhood_search)
                      for cell in neighboring_cells)
end

@inline function particles_in_cell(cell_index, neighborhood_search)
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

# When particles end up with coordinates so big that the cell coordinates
# exceed the range of Int, then `floor(Int, i)` will fail with an InexactError.
# In this case, we can just use typemax(Int), since we can assume that particles
# that far away will not interact with anything, anyway.
# This usually indicates an instability, but we don't want the simulation to crash,
# since adaptive time integration methods may detect the instability and reject the
# time step.
# If we threw an error here, we would prevent the time integration method from
# retrying with a smaller time step, and we would thus crash perfectly fine simulations.
@inline function floor_to_int(i)
    if isnan(i) || i > typemax(Int)
        return typemax(Int)
    elseif i < typemin(Int)
        return typemin(Int)
    end

    return floor(Int, i)
end

# Create a copy of a neighborhood search but with a different search radius
function copy_neighborhood_search(nhs::GridNeighborhoodSearch, search_radius, x, y)
    if nhs.periodic_box === nothing
        search = GridNeighborhoodSearch{ndims(nhs)}(search_radius, nparticles(nhs))
    else
        search = GridNeighborhoodSearch{ndims(nhs)}(search_radius, nparticles(nhs),
                                                    periodic_box_min_corner = nhs.periodic_box.min_corner,
                                                    periodic_box_max_corner = nhs.periodic_box.max_corner)
    end

    # Initialize neighborhood search
    initialize!(search, x, y)

    return search
end
