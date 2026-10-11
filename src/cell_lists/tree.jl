@doc raw"""
    TreeCellList(; min_corner, max_corner, max_level,
                 root_cell_size = maximum(max_corner .- min_corner), capacity = 0,
                 backend = DynamicVectorOfVectors{Int32}, max_points_per_cell = 100)

Cell list for the [`TreeNeighborhoodSearch`](@ref), storing points in the leaves of
a quadtree (2D) or octree (3D) with variable leaf sizes.

The domain is covered by a grid of cubic root cells of size `root_cell_size`.
By default, a single root cell covers the whole domain.
Each root cell is the root of a tree, in which a cell at level ``L``
has the size `root_cell_size` ``/ 2^L``.
The finest possible level is `max_level`.

The tree structure is rebuilt from scratch in every update.
A cell at level ``L`` is split into ``2^d`` children only if the size of its children is
at least as large as the maximum radius of all points in the ``3^d`` block of cells
around this cell at level ``L``, and if it contains more than `capacity` points.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in any future releases.

!!! note "Memory usage"
    The tree data structures are stored densely for all levels. They therefore require
    memory proportional to the number of cells at level `max_level`, which is
    (number of root cells) ``\times 2^{d \cdot \text{max\_level}}``.
    Lists of points are only allocated for non-empty leaves. With the default `backend`,
    `max_points_per_cell` entries are allocated for each of the `n_points` possible
    non-empty leaves (see [`TreeNeighborhoodSearch`](@ref)).

# Keywords
- `min_corner`: Coordinates of the domain corner in negative coordinate directions.
- `max_corner`: Coordinates of the domain corner in positive coordinate directions.
                All points must be inside the domain.
- `max_level`:  The finest level of the trees. Choose this such that
                `root_cell_size` ``/ 2^{\text{max\_level}}`` is not much smaller than the
                smallest search radius.
- `root_cell_size = maximum(max_corner .- min_corner)`: Size of the root cells.
                For elongated domains, a smaller size can be used to obtain a grid of root
                cells and avoid wasting memory for cells outside the domain.
                When the domain is covered by more than one root cell, the search radii
                of all points must not exceed `root_cell_size`.
                For the best performance, use the maximum search radius (if known),
                so that the cell sizes on all levels are aligned with this radius.
- `capacity = 0`: Only split cells that contain more than `capacity` points.
- `backend = DynamicVectorOfVectors{Int32}`: Type of the data structure to store the
    points of each leaf. Can be
    - `Vector{Vector{Int32}}`: Scattered memory, but very memory-efficient.
    - `DynamicVectorOfVectors{Int32}`: Contiguous memory, optimizing cache-hits.
- `max_points_per_cell = 100`: Maximum number of points per leaf. This will be used to
                allocate the `DynamicVectorOfVectors`. It is not used with
                the `Vector{Vector{Int32}}` backend.
                Note that leaves can be larger than the search radius and contain
                more points than the cells of a [`FullGridCellList`](@ref).
"""
struct TreeCellList{NDIMS, ELTYPE, C, LS, MR, NP, LL} <: AbstractCellList
    min_corner     :: SVector{NDIMS, ELTYPE}
    max_corner     :: SVector{NDIMS, ELTYPE}
    root_cell_size :: ELTYPE
    n_roots        :: NTuple{NDIMS, Int}
    max_level      :: Int
    capacity       :: Int
    # Lists of points, one for each non-empty leaf, and an empty list at index 1
    cells::C
    # For each finest cell, the level of the leaf containing it and, at the first finest
    # cell of each leaf, the index in `cells` of the list of this leaf: `(slot << 8) | level`.
    # This way, both are obtained with a single load in the query.
    leaves::LS
    # Pyramids with one entry for each cell at each level, see `pyramid_index`
    max_radius     :: MR  # Maximum radius of all points in this cell
    max_radius_all :: MR  # Maximum radius of all points (one-element vector)
    n_points       :: NP  # Number of points in this cell
    leaf_level     :: LL  # Level of the leaf containing this cell or -1 for split cells
end

function TreeCellList(; min_corner, max_corner, max_level,
                      root_cell_size = maximum(max_corner .- min_corner), capacity = 0,
                      backend = DynamicVectorOfVectors{Int32}, max_points_per_cell = 100)
    NDIMS = length(min_corner)
    if length(max_corner) != NDIMS
        throw(ArgumentError("`min_corner` and `max_corner` must have the same length"))
    end

    if !(NDIMS in (1, 2, 3))
        throw(ArgumentError("`TreeCellList` only supports 1, 2 or 3 dimensions"))
    end

    # The Morton index of a cell inside a root cell must fit into an `Int64`
    if max_level < 0 || NDIMS * max_level > 60
        throw(ArgumentError("`max_level` must be between 0 and $(div(60, NDIMS)) " *
                            "in $(NDIMS)D"))
    end

    ELTYPE = promote_type(eltype(min_corner), eltype(max_corner), typeof(root_cell_size))
    min_corner_ = SVector{NDIMS, ELTYPE}(Tuple(min_corner))
    max_corner_ = SVector{NDIMS, ELTYPE}(Tuple(max_corner))
    root_cell_size_ = convert(ELTYPE, root_cell_size)

    if !(root_cell_size_ > 0)
        throw(ArgumentError("`root_cell_size` must be positive"))
    end

    # Number of root cells needed to cover the domain in each dimension
    n_roots = Tuple(max.(ceil.(Int, (max_corner_ .- min_corner_) ./ root_cell_size_), 1))

    n_cells = level_offset(n_roots, max_level + 1, Val(NDIMS))
    n_finest_cells = prod(n_roots) << (NDIMS * max_level)

    # The lists for the leaves are allocated by `allocate_leaves`, which is called
    # in the constructor of the `TreeNeighborhoodSearch`.
    cells = construct_backend(backend, 1, max_points_per_cell)
    leaves = zeros(Int64, n_finest_cells)
    max_radius = zeros(ELTYPE, n_cells)
    max_radius_all = zeros(ELTYPE, 1)
    n_points = zeros(Int32, n_cells)
    leaf_level = zeros(Int8, n_cells)

    return TreeCellList(min_corner_, max_corner_, root_cell_size_, n_roots, max_level,
                        capacity, cells, leaves, max_radius, max_radius_all,
                        n_points, leaf_level)
end

# Return a copy of `cell_list` sharing the same tree data structures, but with lists
# for up to `n_points` non-empty leaves.
function allocate_leaves(cell_list::TreeCellList, n_points)
    (; cells, leaves) = cell_list

    # There can't be more non-empty leaves than points or finest cells.
    # The first list is always empty and used for all empty leaves.
    n_lists = min(n_points, length(leaves)) + 1
    cells_ = construct_backend(typeof(cells), n_lists, max_inner_length(cells, 100))

    return TreeCellList(cell_list.min_corner, cell_list.max_corner,
                        cell_list.root_cell_size, cell_list.n_roots, cell_list.max_level,
                        cell_list.capacity, cells_, leaves, cell_list.max_radius,
                        cell_list.max_radius_all, cell_list.n_points,
                        cell_list.leaf_level)
end

@inline Base.ndims(::TreeCellList{NDIMS}) where {NDIMS} = NDIMS
@inline Base.eltype(::TreeCellList{<:Any, ELTYPE}) where {ELTYPE} = ELTYPE

function supported_update_strategies(::TreeCellList)
    return (SerialUpdate,)
end

function copy_cell_list(cell_list::TreeCellList)
    (; min_corner, max_corner, max_level, root_cell_size, capacity, cells) = cell_list

    return TreeCellList(; min_corner, max_corner, max_level, root_cell_size, capacity,
                        backend = typeof(cells),
                        max_points_per_cell = max_inner_length(cells, 100))
end

# Each pyramid array stores all cells of level 0, then all cells of level 1, and so on.
# Within each level, cells are sorted by root cell first and then in Morton order
# inside each root cell. See `cell_index`.
# This is the number of cells on all levels coarser than `level`.
@inline function level_offset(n_roots, level, ::Val{NDIMS}) where {NDIMS}
    # Geometric series sum_{l=0}^{level-1} 2^(NDIMS * l)
    n_children = 1 << NDIMS
    return prod(n_roots) * div((1 << (NDIMS * level)) - 1, n_children - 1)
end

@inline function level_offset(cell_list::TreeCellList, level)
    return level_offset(cell_list.n_roots, level, Val(ndims(cell_list)))
end

# Index of the cell with zero-based index `cell_index` at level `level`
# in the pyramid arrays.
@inline function pyramid_index(cell_list, cell_index, level)
    return level_offset(cell_list, level) + cell_index + 1
end

# Size of the cells at level `level`
@inline function cell_size(cell_list::TreeCellList, level)
    (; root_cell_size) = cell_list

    return root_cell_size / (1 << level)
end

# Number of cells at the finest level that are inside one cell at level `level`
@inline function n_finest_cells(cell_list::TreeCellList, level)
    return 1 << (ndims(cell_list) * (cell_list.max_level - level))
end

# Number of cells at level `level` in each dimension
@inline function n_cells_per_dimension(cell_list::TreeCellList, level)
    return cell_list.n_roots .<< level
end

# Zero-based Cartesian coordinates of the cell at level `level` that contains
# the coordinates `coords`. Note that this can be outside of the grid.
@inline function cell_coords(coords, cell_list::TreeCellList, level)
    (; min_corner) = cell_list

    return Tuple(floor_to_int.((coords .- min_corner) ./ cell_size(cell_list, level)))
end

@inline function is_in_grid(cell, cell_list::TreeCellList, level)
    n_cells = n_cells_per_dimension(cell_list, level)

    return all(i -> 0 <= cell[i] < n_cells[i], eachindex(cell))
end

# Zero-based linear index of the cell with zero-based Cartesian coordinates `cell`
# at level `level`.
# Cells are sorted by root cell first and then in Morton order inside each root cell.
# This way, all descendants of a cell are stored contiguously at each level.
# In particular, the parent of cell `i` is `i >> NDIMS` and its children are
# `(i << NDIMS) + 0:(2^NDIMS - 1)`.
@inline function cell_index(cell_list::TreeCellList, cell, level)
    NDIMS = ndims(cell_list)

    return sum(ntuple(dim -> cell_index_part(cell_list, dim, cell[dim], level),
                      Val(NDIMS)))
end

# The linear cell index is the sum of independent parts for each dimension.
# This is the part of dimension `dim` for the zero-based coordinate `coord`
# in this dimension. When looping over a block of cells, these parts only have to be
# computed once for each coordinate in each dimension.
@inline function cell_index_part(cell_list::TreeCellList, dim, coord, level)
    (; n_roots) = cell_list
    NDIMS = ndims(cell_list)

    # Coordinate of the root cell and coordinate within the root cell
    root = coord >> level
    local_coord = coord - (root << level)

    # The root cells are sorted in column-major order
    stride = 1
    for i in 1:(dim - 1)
        stride *= n_roots[i]
    end

    # Within each root cell, interleave the bits of the coordinates (Morton order)
    return ((root * stride) << (NDIMS * level)) |
           (spread_bits(local_coord, Val(NDIMS)) << (dim - 1))
end

# Insert `NDIMS - 1` zero bits between all bits of `x`
@inline spread_bits(x, ::Val{1}) = x
@inline spread_bits(x, ::Val{2}) = spread_bits_2d(x)
@inline spread_bits(x, ::Val{3}) = spread_bits_3d(x)

# Insert one zero bit between all bits of the lower 32 bits of `x`
@inline function spread_bits_2d(x)
    x = UInt64(x) & 0x00000000ffffffff
    x = (x | (x << 16)) & 0x0000ffff0000ffff
    x = (x | (x << 8)) & 0x00ff00ff00ff00ff
    x = (x | (x << 4)) & 0x0f0f0f0f0f0f0f0f
    x = (x | (x << 2)) & 0x3333333333333333
    x = (x | (x << 1)) & 0x5555555555555555

    return Int(x)
end

# Insert two zero bits between all bits of the lower 21 bits of `x`
@inline function spread_bits_3d(x)
    x = UInt64(x) & 0x00000000001fffff
    x = (x | (x << 32)) & 0x001f00000000ffff
    x = (x | (x << 16)) & 0x001f0000ff0000ff
    x = (x | (x << 8)) & 0x100f00f00f00f00f
    x = (x | (x << 4)) & 0x10c30c30c30c30c3
    x = (x | (x << 2)) & 0x1249249249249249

    return Int(x)
end

# Inverse of `cell_index`. Compute the zero-based Cartesian coordinates of the cell with
# zero-based linear index `cell_index` at level `level`.
@inline function cell_coords_from_index(cell_list::TreeCellList, cell_index, level)
    (; n_roots) = cell_list
    NDIMS = ndims(cell_list)

    root_index = cell_index >> (NDIMS * level)
    local_index = cell_index - (root_index << (NDIMS * level))

    # Inverse of the column-major linear index of the root cell
    root = ntuple(Val(NDIMS)) do dim
        stride = 1
        for i in 1:(dim - 1)
            stride *= n_roots[i]
        end

        return rem(div(root_index, stride), n_roots[dim])
    end

    return (root .<< level) .+ morton_coords(local_index, Val(NDIMS))
end

# Inverse of the Morton index of the local cell coordinates in `cell_index_part`
@inline morton_coords(index, ::Val{1}) = (index,)

@inline function morton_coords(index, ::Val{2})
    return (compact_bits_2d(index), compact_bits_2d(index >> 1))
end

@inline function morton_coords(index, ::Val{3})
    return (compact_bits_3d(index), compact_bits_3d(index >> 1),
            compact_bits_3d(index >> 2))
end

# Inverse of `spread_bits_2d`. Extract every second bit of `x`.
@inline function compact_bits_2d(x)
    x = UInt64(x) & 0x5555555555555555
    x = (x | (x >> 1)) & 0x3333333333333333
    x = (x | (x >> 2)) & 0x0f0f0f0f0f0f0f0f
    x = (x | (x >> 4)) & 0x00ff00ff00ff00ff
    x = (x | (x >> 8)) & 0x0000ffff0000ffff
    x = (x | (x >> 16)) & 0x00000000ffffffff

    return Int(x)
end

# Inverse of `spread_bits_3d`. Extract every third bit of `x`.
@inline function compact_bits_3d(x)
    x = UInt64(x) & 0x1249249249249249
    x = (x | (x >> 2)) & 0x10c30c30c30c30c3
    x = (x | (x >> 4)) & 0x100f00f00f00f00f
    x = (x | (x >> 8)) & 0x001f0000ff0000ff
    x = (x | (x >> 16)) & 0x001f00000000ffff
    x = (x | (x >> 32)) & 0x00000000001fffff

    return Int(x)
end

# Zero-based index of the finest cell containing the point with coordinates `coords`.
# Points must be inside the domain.
@inline function finest_cell_index(coords, cell_list::TreeCellList)
    (; min_corner, max_corner, max_level) = cell_list

    # This also catches NaNs
    if !all(min_corner .<= coords .<= max_corner)
        error("particle coordinates are NaN or outside the domain bounds of the cell list")
    end

    # Clamp to the grid to avoid out of bounds cells due to rounding errors
    # and for points exactly on the upper boundary of the grid.
    cell = cell_coords(coords, cell_list, max_level)
    cell_ = clamp.(cell, 0, n_cells_per_dimension(cell_list, max_level) .- 1)

    return cell_index(cell_list, cell_, max_level)
end

# Rebuild the tree from scratch for the points `eachindex_points` with coordinates `coords`
# and radii `radii`.
function rebuild!(cell_list::TreeCellList, coords, radii, eachindex_points)
    (; cells, leaves, max_level, max_radius, n_points) = cell_list
    NDIMS = ndims(cell_list)

    # Only the finest level has to be reset. The coarser levels are computed from it below.
    finest_cells = (level_offset(cell_list, max_level) + 1):length(n_points)
    view(max_radius, finest_cells) .= 0
    view(n_points, finest_cells) .= 0

    for i in eachindex(cells)
        emptyat!(cells, i)
    end

    # Count points and compute the maximum radius of each finest cell
    for point in eachindex_points
        radius = radii[point]
        if !(radius >= 0)
            error("search radii must be non-negative")
        end

        point_coords = extract_svector(coords, Val(NDIMS), point)
        i = pyramid_index(cell_list, finest_cell_index(point_coords, cell_list), max_level)
        n_points[i] += 1
        max_radius[i] = max(max_radius[i], radius)
    end

    # Compute the counts and maximum radii of all coarser cells from their children
    for level in (max_level - 1):-1:0
        for cell in cells_on_level(cell_list, level)
            reduce_children!(cell_list, cell, level)
        end
    end

    check_max_radius(cell_list)

    # Determine the tree structure from the coarsest to the finest level
    for level in 0:max_level
        for cell in CartesianIndices(n_cells_per_dimension(cell_list, level))
            # Convert one-based `CartesianIndex` to zero-based tuple
            split_cell!(cell_list, Tuple(cell) .- 1, level)
        end
    end

    # Assign a list to each non-empty leaf. Leaves are identified by their first
    # finest cell. All other finest cells and empty leaves use the empty list 1.
    n_lists = 1
    for finest_cell in 0:(length(leaves) - 1)
        level = Int(cell_list.leaf_level[pyramid_index(cell_list, finest_cell, max_level)])
        shift = NDIMS * (max_level - level)
        leaf_index = finest_cell >> shift
        is_first_cell = (leaf_index << shift) == finest_cell

        slot = 1
        if is_first_cell && n_points[pyramid_index(cell_list, leaf_index, level)] > 0
            n_lists += 1
            slot = n_lists
        end
        leaves[finest_cell + 1] = (slot << 8) | level
    end

    if n_lists > length(cells)
        error("the number of points exceeds `n_points` of the neighborhood search")
    end

    # Add each point to its leaf
    for point in eachindex_points
        point_coords = extract_svector(coords, Val(NDIMS), point)
        finest_cell = finest_cell_index(point_coords, cell_list)
        pushat!(cells, leaf_slot(cell_list, leaf_first_cell(cell_list, finest_cell)), point)
    end

    return cell_list
end

# Zero-based indices of all cells at level `level`
@inline function cells_on_level(cell_list::TreeCellList, level)
    return 0:(level_offset(cell_list, level + 1) - level_offset(cell_list, level) - 1)
end

# Compute the number of points and the maximum radius of a cell from its children
@inline function reduce_children!(cell_list, cell, level)
    (; max_radius, n_points) = cell_list
    NDIMS = ndims(cell_list)

    i = pyramid_index(cell_list, cell, level)
    first_child = pyramid_index(cell_list, cell << NDIMS, level + 1)

    max_radius_ = zero(eltype(max_radius))
    n_points_ = zero(eltype(n_points))
    for child in first_child:(first_child + (1 << NDIMS) - 1)
        max_radius_ = max(max_radius_, max_radius[child])
        n_points_ += n_points[child]
    end

    max_radius[i] = max_radius_
    n_points[i] = n_points_

    return cell_list
end

function check_max_radius(cell_list)
    (; max_radius, max_radius_all, n_roots, root_cell_size) = cell_list

    # The maximum radius of all points is the maximum of all root cells
    max_radius_all_ = maximum(view(max_radius, 1:prod(n_roots)),
                              init = zero(eltype(max_radius)))
    fill!(max_radius_all, max_radius_all_)

    # With a single root cell, the 3^d block around the root cell contains all points.
    # With multiple root cells, we can only guarantee to find all neighbors when
    # all radii are smaller than the root cells.
    if prod(n_roots) > 1 && max_radius_all_ > root_cell_size
        error("the search radii must not exceed the `root_cell_size` of the cell list " *
              "when the domain is covered by multiple root cells")
    end
end

# Decide whether the cell with zero-based Cartesian coordinates `cell` at level `level`
# is split, and store the level of the leaf containing this cell in `leaf_level`.
# Note that this requires the parent level to be processed already.
@inline function split_cell!(cell_list, cell, level)
    (; leaf_level, n_points, capacity, max_level) = cell_list
    NDIMS = ndims(cell_list)

    cell_index_ = cell_index(cell_list, cell, level)
    i = pyramid_index(cell_list, cell_index_, level)

    if level > 0
        parent_leaf_level = leaf_level[pyramid_index(cell_list, cell_index_ >> NDIMS,
                                                     level - 1)]
        if parent_leaf_level >= 0
            # This cell is inside a coarser leaf
            leaf_level[i] = parent_leaf_level
            return cell_list
        end
    end

    # This cell is a node of the tree. Split it if possible.
    if level < max_level && n_points[i] > capacity && can_split(cell_list, cell, level)
        leaf_level[i] = -1
    else
        leaf_level[i] = level
    end

    return cell_list
end

# A cell can be split if its children are at least as large as the maximum radius of all
# points in the 3^d block of cells around this cell at the same level.
# This guarantees that for any point with a search radius `h_i` and a leaf
# at level `L` with `h_i <= cell_size(L)`, all neighbors `j` with
# `distance <= max(h_i, h_j)` are in the 3^d block around this leaf at level `L`.
# See the documentation of `TreeNeighborhoodSearch` for a proof.
@inline function can_split(cell_list, cell, level)
    (; max_radius) = cell_list
    NDIMS = ndims(cell_list)

    child_size = cell_size(cell_list, level + 1)

    for neighbor_cell_ in CartesianIndices(ntuple(i -> (cell[i] - 1):(cell[i] + 1),
                                Val(NDIMS)))
        neighbor_cell = Tuple(neighbor_cell_)
        is_in_grid(neighbor_cell, cell_list, level) || continue

        neighbor_index = cell_index(cell_list, neighbor_cell, level)
        if max_radius[pyramid_index(cell_list, neighbor_index, level)] > child_size
            return false
        end
    end

    return true
end

# Level of the leaf containing the finest cell with zero-based index `finest_cell`
@propagate_inbounds function leaf_level(cell_list::TreeCellList, finest_cell)
    return Int(cell_list.leaves[finest_cell + 1] & 0xff)
end

# Zero-based index of the first finest cell of the leaf containing the finest cell
# with zero-based index `finest_cell`
@propagate_inbounds function leaf_first_cell(cell_list::TreeCellList, finest_cell)
    shift = ndims(cell_list) * (cell_list.max_level - leaf_level(cell_list, finest_cell))

    return (finest_cell >> shift) << shift
end

# Index in `cell_list.cells` of the list containing the points of the leaf with the
# first finest cell `first_finest_cell` (zero-based index)
@propagate_inbounds function leaf_slot(cell_list::TreeCellList, first_finest_cell)
    return cell_list.leaves[first_finest_cell + 1] >> 8
end

# The points of the leaf with the first finest cell `first_finest_cell`
@propagate_inbounds function points_in_leaf(cell_list::TreeCellList, first_finest_cell)
    return cell_list.cells[leaf_slot(cell_list, first_finest_cell)]
end
