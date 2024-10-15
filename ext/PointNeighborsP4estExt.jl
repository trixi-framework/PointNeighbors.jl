module PointNeighborsP4estExt

using PointNeighbors: PointNeighbors, SVector, DynamicVectorOfVectors,
                      ParallelUpdate, SemiParallelUpdate, SerialUpdate,
                      AbstractCellList, pushat!, pushat_atomic!, emptyat!, deleteatat!
using P4estTypes: P4estTypes

"""
    P4estCellList(; min_corner, max_corner, search_radius = 0.0,
                     periodicity = false, backend = DynamicVectorOfVectors{Int32},
                     max_points_per_cell = 100)

A simple cell list implementation where each (empty or non-empty) cell of a rectangular
(axis-aligned) domain is assigned a list of points.
This cell list only works when all points are inside the specified domain at all times.

Only set `min_corner` and `max_corner` and use the default values for the other arguments
to create an empty "template" cell list that can be used to create an empty "template"
neighborhood search.
See [`copy_neighborhood_search`](@ref) for more details.

# Keywords
- `min_corner`: Coordinates of the domain corner in negative coordinate directions.
- `max_corner`: Coordinates of the domain corner in positive coordinate directions.
- `search_radius = 0.0`: Search radius of the neighborhood search, which will determine the
                         cell size. Use the default of `0.0` to create a template (see above).
- `periodicity = false`: Set to `true` when using a [`PeriodicBox`](@ref) with the
                         neighborhood search. When using [`copy_neighborhood_search`](@ref),
                         this option can be ignored an will be set automatically depending
                         on the periodicity of the neighborhood search.
- `backend = DynamicVectorOfVectors{Int32}`: Type of the data structure to store the actual
    cell lists. Can be
    - `Vector{Vector{Int32}}`: Scattered memory, but very memory-efficient.
    - `DynamicVectorOfVectors{Int32}`: Contiguous memory, optimizing cache-hits.
- `max_points_per_cell = 100`: Maximum number of points per cell. This will be used to
                               allocate the `DynamicVectorOfVectors`. It is not used with
                               the `Vector{Vector{Int32}}` backend.
"""
struct P4estCellList{C, CI, P, NC, MINC, MAXC} <: AbstractCellList
    cells          :: C
    cell_indices   :: CI
    neighbor_cells :: NC
    p4est          :: P
    min_corner     :: MINC
    max_corner     :: MAXC
end

function PointNeighbors.supported_update_strategies(::P4estCellList{<:DynamicVectorOfVectors})
    return (ParallelUpdate, SemiParallelUpdate, SerialUpdate)
end

PointNeighbors.supported_update_strategies(::P4estCellList) = (SemiParallelUpdate, SerialUpdate)

function PointNeighbors.P4estCellList(; min_corner, max_corner, search_radius = 0.0,
                       backend = DynamicVectorOfVectors{Int32},
                       max_points_per_cell = 100)
    # Pad domain to avoid 0 in cell indices due to rounding errors.
    # We can't just use `eps()`, as one might use lower precision types.
    # This padding is safe, and will give us one more layer of cells in the worst case.
    min_corner = SVector(Tuple(min_corner .- 1e-3 * search_radius))
    max_corner = SVector(Tuple(max_corner .+ 1e-3 * search_radius))

    if search_radius < eps()
        # Create an empty "template" cell list to be used with `copy_cell_list`
        cells = construct_backend(backend, 0, 0)
        cell_indices = nothing
        neighbors = nothing
        p4est = nothing
    else
        # Note that we don't shift everything so that the first cell starts at `min_corner`.
        # The first cell is the cell containing `min_corner`, so we need to add one layer
        # in order for `max_corner` to be inside a cell.
        n_cells_per_dimension = ceil.(Int, (max_corner .- min_corner) ./ search_radius) .+ 1

        connectivity = P4estTypes.brick(Tuple(n_cells_per_dimension))
        p4est = P4estTypes.pxest(connectivity)

        cell_indices = find_cell_indices(p4est, n_cells_per_dimension)
        neighbors = find_neighbors(p4est)

        cells = construct_backend(backend, n_cells_per_dimension, max_points_per_cell)
    end

    return P4estCellList(cells, cell_indices, neighbors, p4est, min_corner, max_corner)
end

function construct_backend(::Type{Vector{Vector{T}}}, size, max_points_per_cell) where {T}
    return [T[] for _ in 1:prod(size)]
end

function construct_backend(::Type{DynamicVectorOfVectors{T}}, size,
                           max_points_per_cell) where {T}
    cells = DynamicVectorOfVectors{T}(max_outer_length = prod(size),
                                      max_inner_length = max_points_per_cell)
    resize!(cells, prod(size))

    return cells
end

# When `typeof(cell_list.cells)` is passed, we don't pass the type
# `DynamicVectorOfVectors{T}`, but a type `DynamicVectorOfVectors{T1, T2, T3, T4}`.
# While `A{T} <: A{T1, T2}`, this doesn't hold for the types.
# `Type{A{T}} <: Type{A{T1, T2}}` is NOT true.
function construct_backend(::Type{DynamicVectorOfVectors{T1, T2, T3, T4}}, size,
                           max_points_per_cell) where {T1, T2, T3, T4}
    return construct_backend(DynamicVectorOfVectors{T1}, size, max_points_per_cell)
end

function find_cell_indices(p4est, n_cells_per_dimension)
    vertices = P4estTypes.unsafe_vertices(p4est.connectivity)
    cartesian_indices = CartesianIndices((n_cells_per_dimension..., 1))
    vertex_indices = map(i -> findfirst(==(Tuple(i) .- 1), vertices),
                         collect(cartesian_indices))
    trees_to_first_vertex = first.(P4estTypes.unsafe_trees(p4est.connectivity)) .+ 1
    cell_indices = map(i -> findfirst(==(i), trees_to_first_vertex), vertex_indices)

    return cell_indices
end

# Load the ith element (1-indexed) of an sc array of type T as PointerWrapper
function load_pointerwrapper_sc(::Type{T}, sc_array::P4estTypes.PointerWrapper{P4estTypes.sc_array},
                                i::Integer = 1) where {T}
    return P4estTypes.PointerWrapper(T, pointer(sc_array.array) + (i - 1) * sizeof(T))
end

function find_neighbors_iter_corner(::Type{T}) where {T}
    function f(info_ptr, user_data_ptr)
        info = P4estTypes.PointerWrapper(info_ptr)
        n = info.sides.elem_count[]

        user_data = (unsafe_pointer_to_objref(user_data_ptr)::Base.RefValue{T})[]
        (; neighbors, neighbors_mpi, buffer, buffer_ghost, proc_offsets) = user_data

        n_local = 0
        n_ghost = 0
        for i in 1:n
            side = load_pointerwrapper_sc(P4estTypes.p4est_iter_corner_side_t,
                                          info.sides, i)
            tree = load_pointerwrapper_sc(P4estTypes.p4est_tree_t, info.p4est.trees,
                                          side.treeid[] + 1)
            offset = tree.quadrants_offset[]
            local_quad_id = side.quadid[]
            cell_id = offset + local_quad_id + 1

            if side.is_ghost[] == 0
                n_local += 1
                buffer[n_local] = cell_id
            else
                n_ghost += 1
                ghost_id = local_quad_id

                # MPI ranks are 0-based
                rank = searchsortedlast(proc_offsets, ghost_id) - 1
                local_id = side.quad.p.piggy3.local_num[] + 1
                buffer_ghost[n_ghost] = (rank, local_id)
            end
        end

        # Add to neighbor lists
        for i in 1:n_local
            cell = buffer[i]
            for j in 1:n_local
                neighbor = buffer[j]
                if !(neighbor in neighbors[cell])
                    pushat!(neighbors, cell, neighbor)
                end
            end

            for j in 1:n_ghost
                neighbor = buffer_ghost[j]
                if !(neighbor in neighbors_mpi[cell])
                    pushat!(neighbors_mpi, cell, neighbor)
                end
            end
        end

        return nothing
    end

    return f
end

@generated function cfunction(::typeof(find_neighbors_iter_corner), ::Val{2}, ::T) where {T}
    f = find_neighbors_iter_corner(T)
    quote
       @cfunction($f, Cvoid,
               (Ptr{P4estTypes.p4est_iter_corner_info_t}, Ptr{Cvoid}))
    end
end

function find_neighbors(p4est)
    n_cells = P4estTypes.lengthoflocalquadrants(p4est)
    neighbors = DynamicVectorOfVectors{Int32}(max_outer_length = n_cells,
                                              max_inner_length = 3^2)
    mpi_neighbors = DynamicVectorOfVectors{Tuple{Int32, Int32}}(max_outer_length = n_cells,
                                                                max_inner_length = 3^2)

    # Buffer to store the (at most) 8 cells adjacent to a corner
    buffer = zeros(Int32, 8)
    buffer_ghost = fill((0, 0), 8)

    find_neighbors!(neighbors, mpi_neighbors, p4est, buffer, buffer_ghost)

    return neighbors, mpi_neighbors
end

@inline function find_neighbors!(neighbors, neighbors_mpi, p4est, buffer, buffer_ghost)
    n_cells = P4estTypes.lengthoflocalquadrants(p4est)

    empty!(neighbors)
    resize!(neighbors, n_cells)
    empty!(neighbors_mpi)
    resize!(neighbors_mpi, n_cells)

    ghost_layer = P4estTypes.ghostlayer(p4est, connection=P4estTypes.CONNECT_CORNER(Val(4)))
    proc_offsets = P4estTypes.unsafe_proc_offsets(ghost_layer)

    user_data = (; neighbors, neighbors_mpi, buffer, buffer_ghost, proc_offsets)

    # Let `p4est` iterate over all corner and call find_neighbors_iter_corner
    iter_corner_c = cfunction(find_neighbors_iter_corner, Val(2), user_data)

    # Compute the neighbors of each cell
    GC.@preserve user_data begin
        P4estTypes.p4est_iterate(p4est,
                      ghost_layer, # ghost_layer
                      pointer_from_objref(Ref(user_data)), # user_data
                      C_NULL, # iter_volume
                      C_NULL, # iter_face
                      iter_corner_c) # iter_corner
    end

    return neighbors
end

@inline function PointNeighbors.neighboring_cells(cell, neighborhood_search::P4estCellList)
    return neighborhood_search.neighbor_cells[neighborhood_search.cell_indices[cell...]]
end

@inline function PointNeighbors.cell_coords(coords, periodic_box::Nothing, cell_list::P4estCellList,
                             cell_size)
    (; min_corner) = cell_list

    # Subtract `min_corner` to offset coordinates so that the min corner of the grid
    # corresponds to the (1, 1, 1) cell.
    # Note that we use `min_corner == periodic_box.min_corner`, so we don't have to handle
    # periodic boxes differently, as they also use 1-based indexing.
    return Tuple(PointNeighbors.floor_to_int.((coords .- min_corner) ./ cell_size)) .+ 1
end

function Base.empty!(cell_list::P4estCellList)
    (; cells) = cell_list

    # `Base.empty!.(cells)`, but for all backends
    for i in eachindex(cells)
        emptyat!(cells, i)
    end

    return cell_list
end

function Base.empty!(cell_list::P4estCellList{Nothing})
    # This is an empty "template" cell list to be used with `copy_cell_list`
    error("`search_radius` is not defined for this cell list")
end

function PointNeighbors.push_cell!(cell_list::P4estCellList, cell, particle)
    (; cells) = cell_list

    # `push!(cell_list[cell], particle)`, but for all backends
    pushat!(cells, PointNeighbors.cell_index(cell_list, cell), particle)

    return cell_list
end

function PointNeighbors.push_cell!(cell_list::P4estCellList{Nothing}, cell, particle)
    # This is an empty "template" cell list to be used with `copy_cell_list`
    error("`search_radius` is not defined for this cell list")
end

@inline function PointNeighbors.push_cell_atomic!(cell_list::P4estCellList, cell, particle)
    (; cells) = cell_list

    # `push!(cell_list[cell], particle)`, but for all backends.
    # The atomic version of `pushat!` uses atomics to avoid race conditions when `pushat!`
    # is used in a parallel loop.
    pushat_atomic!(cells, PointNeighbors.cell_index(cell_list, cell), particle)

    return cell_list
end

function PointNeighbors.deleteat_cell!(cell_list::P4estCellList, cell, i)
    (; cells) = cell_list

    # `deleteat!(cell_list[cell], i)`, but for all backends
    deleteatat!(cells, PointNeighbors.cell_index(cell_list, cell), i)
end

@inline PointNeighbors.each_cell_index(cell_list::P4estCellList) = eachindex(cell_list.cells)

function PointNeighbors.each_cell_index(cell_list::P4estCellList{Nothing})
    # This is an empty "template" cell list to be used with `copy_cell_list`
    error("`search_radius` is not defined for this cell list")
end

@inline function PointNeighbors.cell_index(cell_list::P4estCellList, cell::Tuple)
    (; cell_indices) = cell_list

    return cell_indices[cell...]
end

@inline PointNeighbors.cell_index(::P4estCellList, cell::Integer) = cell

@inline function Base.getindex(cell_list::P4estCellList, cell)
    (; cells) = cell_list

    return cells[PointNeighbors.cell_index(cell_list, cell)]
end

@inline function PointNeighbors.is_correct_cell(cell_list::P4estCellList, cell_coords, cell_index_)
    return PointNeighbors.cell_index(cell_list, cell_coords) == cell_index_
end

@inline PointNeighbors.index_type(::P4estCellList) = Int32

function PointNeighbors.copy_cell_list(cell_list::P4estCellList, search_radius, periodic_box)
    (; min_corner, max_corner) = cell_list

    return PointNeighbors.P4estCellList(; min_corner, max_corner, search_radius,
                                        backend = typeof(cell_list.cells))
end

end # module
