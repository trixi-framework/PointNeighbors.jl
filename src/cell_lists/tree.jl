@doc raw"""
    TreeCellList(; min_corner, max_corner, root_cell_size = zero(eltype(min_corner)),
                 max_level = nothing, capacity = 0,
                 backend = DynamicVectorOfVectors{Int32}, max_points_per_cell = 100)

Cell list for the [`TreeNeighborhoodSearch`](@ref), storing points in the leaves of
a quadtree (2D) or octree (3D) with variable leaf sizes.

The domain is covered by a grid of cubic root cells of size `root_cell_size`.
With periodicity, the root cells tile the periodic box exactly, so they can be slightly
larger than `root_cell_size` and differ in size between the dimensions.
Each root cell is the root of a tree, in which a cell at level ``L``
has the size of the root cell ``/ 2^L``.
The finest possible level is `max_level`.
The cells on each level are sorted by root cell first and then in Morton order inside
each root cell, so that all descendants of a cell are stored contiguously.

The tree structure is rebuilt from scratch in every update.
A cell at level ``L`` is split into ``2^d`` children only if the size of its children is
at least as large as the maximum radius of all points in the ``3^d`` block of cells
around this cell at level ``L``, and if it contains more than `capacity` points.
With a uniform search radius, all leaves containing points are therefore at the finest
level with cells that are at least as large as the search radius.

Only set `min_corner` and `max_corner` (and optionally `max_level`) and use the default
values for the other arguments to create an empty "template" cell list that can be used
to create an empty "template" neighborhood search.
When copying the template neighborhood search with [`copy_neighborhood_search`](@ref),
the size of the root cells is the search radius.

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
- `root_cell_size`: Size of the root cells. The search radii of all points and all query
                radii must not exceed this size. Root cells of the size of the maximum
                search radius are the most efficient choice.
                Use the default of zero to create a template (see above).
- `max_level = nothing`: The finest level of the trees. Choose this such that
                `root_cell_size` ``/ 2^{\text{max\_level}}`` is not much smaller than the
                smallest search radius. By default, this is the finest level with cells
                that are at least as large as the `search_radius` of the
                [`TreeNeighborhoodSearch`](@ref), which is the optimal choice for
                a uniform search radius.
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
    min_corner::SVector{NDIMS, ELTYPE}
    max_corner::SVector{NDIMS, ELTYPE}
    # Size of the root cells requested by the user
    root_cell_size::ELTYPE
    # The grid of root cells, see `root_grid`
    grid_min_corner::SVector{NDIMS, ELTYPE}
    # Actual size of the root cells in each dimension. Without periodicity, this is
    # `root_cell_size` in every dimension. With periodicity, it can be larger.
    actual_root_cell_size::SVector{NDIMS, ELTYPE}
    n_roots::NTuple{NDIMS, Int}
    # The finest level, or -1 if it is chosen by the neighborhood search
    # in `build_tree`
    max_level::Int
    capacity::Int
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

function TreeCellList(; min_corner, max_corner,
                      root_cell_size = zero(eltype(min_corner)), max_level = nothing,
                      capacity = 0, backend = DynamicVectorOfVectors{Int32},
                      max_points_per_cell = 100)
    NDIMS = length(min_corner)
    if length(max_corner) != NDIMS
        throw(ArgumentError("`min_corner` and `max_corner` must have the same length"))
    end

    if !(NDIMS in (1, 2, 3))
        throw(ArgumentError("`TreeCellList` only supports 1, 2 or 3 dimensions"))
    end

    # This also catches NaNs, as opposed to `root_cell_size > 0`.
    if !(root_cell_size >= 0)
        throw(ArgumentError("`root_cell_size` must not be negative or NaN"))
    end

    if eltype(min_corner) <: Integer
        throw(ArgumentError("`min_corner` and `max_corner` cannot be of integer type"))
    end

    # The Morton index of a cell inside a root cell must fit into an `Int64`
    if !isnothing(max_level) && (max_level < 0 || NDIMS * max_level > 60)
        throw(ArgumentError("`max_level` must be between 0 and $(div(60, NDIMS)) " *
                            "in $(NDIMS)D"))
    end

    ELTYPE = promote_type(eltype(min_corner), eltype(max_corner), typeof(root_cell_size))
    min_corner_ = SVector{NDIMS, ELTYPE}(Tuple(min_corner))
    max_corner_ = SVector{NDIMS, ELTYPE}(Tuple(max_corner))
    root_cell_size_ = convert(ELTYPE, root_cell_size)

    (grid_min_corner, actual_root_cell_size,
     n_roots) = root_grid(min_corner_, max_corner_, root_cell_size_, nothing)

    # The tree data structures are allocated by `build_tree`, which is called
    # in the constructor of the `TreeNeighborhoodSearch`.
    cells = construct_backend(backend, 0, max_points_per_cell)
    leaves = zeros(Int64, 0)
    max_radius = zeros(ELTYPE, 0)
    n_points = zeros(Int32, 0)
    leaf_level = zeros(Int8, 0)

    return TreeCellList(min_corner_, max_corner_, root_cell_size_, grid_min_corner,
                        actual_root_cell_size, n_roots, something(max_level, -1), capacity,
                        cells, leaves, max_radius, zeros(ELTYPE, 1), n_points, leaf_level)
end

# The grid of root cells as `(grid_min_corner, actual_root_cell_size, n_roots)`, where
# `grid_min_corner` is the corner of the grid in negative coordinate directions
# and `actual_root_cell_size` is the actual size of the root cells in each dimension.
#
# Without periodicity, the grid has one padding layer of root cells around the domain.
# Then, the 3^d block of cells around each cell inside the domain is inside the grid
# on every level, and we don't need to clip this block to the grid in the query.
function root_grid(min_corner, max_corner, root_cell_size, periodic_box::Nothing)
    NDIMS = length(min_corner)

    if iszero(root_cell_size)
        # This is an empty "template" cell list to be used with `copy_cell_list`
        return min_corner, zero(min_corner), ntuple(_ -> 0, NDIMS)
    end

    n_roots_domain = max.(ceil.(Int, (max_corner .- min_corner) ./ root_cell_size), 1)
    n_roots = Tuple(n_roots_domain .+ 2)
    grid_min_corner = min_corner .- root_cell_size
    actual_root_cell_size = root_cell_size .+ zero(min_corner)

    return grid_min_corner, actual_root_cell_size, n_roots
end

# With periodicity, enlarge root cells as needed to tile the periodic box exactly.
# Cells on opposite boundaries are neighbors.
function root_grid(min_corner, max_corner, root_cell_size, periodic_box)
    if !isapprox(min_corner, periodic_box.min_corner) ||
       !isapprox(max_corner, periodic_box.max_corner)
        throw(ArgumentError("the domain of the `TreeCellList` must be the periodic box"))
    end

    # Allow a small tolerance to avoid inefficient larger cells due to machine
    # rounding errors, as in the `GridNeighborhoodSearch`
    n_roots = Tuple(floor.(Int, (periodic_box.size .+ 10eps()) ./ root_cell_size))

    # With at least 3 root cells in each dimension, the 3^d block of cells around a cell
    # can never contain multiple periodic images of the same cell
    if any(<(3), n_roots)
        throw(ArgumentError("the `TreeCellList` needs at least 3 root cells in each " *
                            "dimension when used with periodicity. " *
                            "Please use a smaller `root_cell_size`."))
    end

    # Convert to the element type of the cell list, which can differ from the element type
    # of the periodic box.
    ELTYPE = eltype(min_corner)
    grid_min_corner = convert.(ELTYPE, periodic_box.min_corner)
    actual_root_cell_size = convert.(ELTYPE, periodic_box.size ./ n_roots)

    return grid_min_corner, actual_root_cell_size, n_roots
end

# Return a copy of `cell_list` with the grid of root cells for `periodic_box`
# and the tree data structures allocated, including lists for up to `n_points` non-empty
# leaves.
# Without a given `max_level`, the finest level is the finest level with cells that are
# at least as large as `search_radius`.
function build_tree(cell_list::TreeCellList, search_radius, n_points_, periodic_box)
    (; min_corner, max_corner, root_cell_size, capacity, cells) = cell_list
    NDIMS = ndims(cell_list)
    ELTYPE = eltype(cell_list)

    # This also catches NaNs, as opposed to `search_radius > 0`.
    if !(search_radius >= 0)
        throw(ArgumentError("the search radius must not be negative or NaN"))
    end

    if iszero(root_cell_size)
        if !iszero(search_radius)
            throw(ArgumentError("the `root_cell_size` of the `TreeCellList` must be " *
                                "set, unless creating a template neighborhood search"))
        end

        # This is an empty "template" neighborhood search to be used with
        # `copy_neighborhood_search`
        return cell_list
    end

    # The 3^d block of cells around a leaf only contains all neighbors if the cells
    # at level 0 are at least as large as the search radius
    if search_radius > root_cell_size
        throw(ArgumentError("the search radius must not exceed the `root_cell_size` of " *
                            "the `TreeCellList`"))
    end

    (grid_min_corner, actual_root_cell_size,
     n_roots) = root_grid(min_corner, max_corner, root_cell_size, periodic_box)

    max_level = cell_list.max_level
    if max_level < 0
        # The Morton index of a cell inside a root cell must fit into an `Int64`
        max_level = 0
        while max_level < div(60, NDIMS) && search_radius > 0 &&
                  minimum(actual_root_cell_size) / (1 << (max_level + 1)) >= search_radius
            max_level += 1
        end
    end

    n_cells = level_offset(n_roots, max_level + 1, Val(NDIMS))
    n_finest_cells = prod(n_roots) << (NDIMS * max_level)

    # There can't be more non-empty leaves than points or finest cells.
    # The first list is always empty and used for all empty leaves.
    n_lists = min(n_points_, n_finest_cells) + 1
    cells_ = construct_backend(typeof(cells), n_lists, max_inner_length(cells, 100))
    leaves = zeros(Int64, n_finest_cells)
    max_radius = zeros(ELTYPE, n_cells)
    max_radius_all = zeros(ELTYPE, 1)
    n_points = zeros(Int32, n_cells)
    leaf_level = zeros(Int8, n_cells)

    return TreeCellList(min_corner, max_corner, root_cell_size, grid_min_corner,
                        actual_root_cell_size, n_roots, max_level, capacity, cells_,
                        leaves, max_radius, max_radius_all, n_points, leaf_level)
end

@inline Base.ndims(::TreeCellList{NDIMS}) where {NDIMS} = NDIMS
@inline Base.eltype(::TreeCellList{<:Any, ELTYPE}) where {ELTYPE} = ELTYPE

function supported_update_strategies(::TreeCellList)
    return (SerialUpdate,)
end

# Return an empty cell list with the same configuration as `cell_list`, but with root cells
# of the size of the search radius `search_radius`.
function copy_cell_list(cell_list::TreeCellList, search_radius)
    (; min_corner, max_corner, capacity, cells) = cell_list

    # Keep `max_level` only if it was set explicitly
    max_level = cell_list.max_level < 0 ? nothing : cell_list.max_level

    return TreeCellList(; min_corner, max_corner, root_cell_size = search_radius,
                        max_level, capacity, backend = typeof(cells),
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

# Size of the cells at level `level` in each dimension
@inline function cell_size(cell_list::TreeCellList, level)
    (; actual_root_cell_size) = cell_list

    return actual_root_cell_size ./ (1 << level)
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
    (; grid_min_corner) = cell_list

    return Tuple(floor_to_int.((coords .- grid_min_corner) ./ cell_size(cell_list, level)))
end

# With periodicity, map a cell outside of the grid to the corresponding cell
# inside the grid
@inline periodic_cell(cell, n_cells, periodic_box::Nothing) = cell
@inline periodic_cell(cell, n_cells, periodic_box) = mod.(cell, n_cells)

# ==========================================================================================
# Linear cell index
# ==========================================================================================
#
# The cells of each level are sorted by root cell first and then in Morton order (Z-order)
# inside each root cell. For example, in 2D with 2 x 1 root cells at level 2, so that each
# root cell contains 4 x 4 cells, the zero-based linear indices of the cells are
#
#              root cell 0         root cell 1
#     y = 3 | 10  11  14  15 | 26  27  30  31 |
#     y = 2 |  8   9  12  13 | 24  25  28  29 |
#     y = 1 |  2   3   6   7 | 18  19  22  23 |
#     y = 0 |  0   1   4   5 | 16  17  20  21 |
#            ---------------------------------
#       x =    0   1   2   3    4   5   6   7
#
# The Morton index of a cell inside its root cell interleaves the bits of the local
# coordinates of the cell inside the root cell: bit `i` of the local coordinate in dimension
# `dim` is bit `NDIMS * i + dim - 1` of the Morton index.
# For example, the local coordinates (1, 2) = (0b01, 0b10) yield the Morton index
# 0b1001 = 9.
# This is handled by the function `local_index_part` below.
#
# Since the bits of different dimensions don't overlap, the linear index is a sum of
# independent parts for each dimension (see `cell_index_part`):
#
#     part = root coordinate * root_stride + local_index_part(...)
#
# For example, the cell (5, 2) has the local coordinates (1, 2) in the root cell (1, 0).
# The root strides are 16 in x and 2 * 16 = 32 in y, so the parts are
#
#     x: 1 * 16 + 0b01   = 17,
#     y: 0 * 32 + 0b1000 = 8,
#
# and the linear index is 17 + 8 = 25.
# This is handled by the function `cell_index_part` below.

# Zero-based linear index of the cell with zero-based Cartesian coordinates `cell`
# at level `level` (see above).
# This way, all descendants of a cell are stored contiguously at each level.
# In particular, the parent of cell `i` is `i >> NDIMS` and its children are
# `(i << NDIMS) .+ (0:(2^NDIMS - 1))`.
@inline function cell_index(cell_list::TreeCellList{NDIMS}, cell::NTuple{NDIMS},
                            level) where {NDIMS}
    # Since `cell` is guaranteed to be of size `NDIMS`, we can use `@inbounds` here.
    return sum(ntuple(@inline(dim -> @inbounds cell_index_part(cell_list, dim, cell[dim],
                                                               level)), Val(NDIMS)))
end

# The part of the linear cell index (see above) of dimension `dim` for the zero-based
# coordinate `coord` in this dimension at level `level`.
@inline function cell_index_part(cell_list::TreeCellList, dim, coord, level)
    # Coordinate of the root cell and coordinate within the root cell.
    root = coord >> level
    local_coord = coord - (root << level)

    return root * root_stride(cell_list, dim, level) +
           local_index_part(local_coord, dim, Val(ndims(cell_list)))
end

# Difference between the linear indices of neighboring root cells in dimension `dim`
# at level `level`. The root cells are sorted in column-major order, and each root cell
# contains `2^(NDIMS * level)` cells.
@inline function root_stride(cell_list, dim, level)
    (; n_roots) = cell_list

    stride = 1
    for i in 1:(dim - 1)
        stride *= n_roots[i]
    end

    return stride << (ndims(cell_list) * level)
end

# The part of the Morton index (see above) of the local coordinate `local_coord`
# in dimension `dim`.
@inline function local_index_part(local_coord, dim, ndims::Val)
    return spread_bits(local_coord, ndims) << (dim - 1)
end

# Insert `NDIMS - 1` zero bits between all bits of `x`, so that bit `i` of `x` becomes
# bit `NDIMS * i`. For example, `0b111` becomes `0b10101` in 2D and `0b1001001` in 3D.
# This is equivalent to the following loop, but requires much fewer operations:
#
#     result = 0
#     for i in 0:(n_bits - 1)
#         result |= ((x >> i) & 1) << (NDIMS * i)
#     end
#
# Instead of moving one bit at a time, the bits are moved in groups. First, the upper half
# of the bits is moved to its final region. Then, the upper half of each half is moved, and
# so on, until each bit is at its final position. In each step, `x | (x << shift)` copies
# all bits, and the mask keeps only the copies at their new positions.
# See https://graphics.stanford.edu/~seander/bithacks.html#InterleaveBMN.
@inline spread_bits(x, ::Val{1}) = x
@inline spread_bits(x, ::Val{2}) = spread_bits_2d(x)
@inline spread_bits(x, ::Val{3}) = spread_bits_3d(x)

# Insert one zero bit between all bits of the lower 32 bits of `x`.
@inline function spread_bits_2d(x)
    x = UInt64(x) & 0x00000000ffffffff
    x = (x | (x << 16)) & 0x0000ffff0000ffff
    x = (x | (x << 8)) & 0x00ff00ff00ff00ff
    x = (x | (x << 4)) & 0x0f0f0f0f0f0f0f0f
    x = (x | (x << 2)) & 0x3333333333333333
    x = (x | (x << 1)) & 0x5555555555555555

    # This is just one LLVM operation as opposed to `Int(x)`.
    return x % Int
end

# Insert two zero bits between all bits of the lower 21 bits of `x`.
@inline function spread_bits_3d(x)
    x = UInt64(x) & 0x00000000001fffff
    x = (x | (x << 32)) & 0x001f00000000ffff
    x = (x | (x << 16)) & 0x001f0000ff0000ff
    x = (x | (x << 8)) & 0x100f00f00f00f00f
    x = (x | (x << 4)) & 0x10c30c30c30c30c3
    x = (x | (x << 2)) & 0x1249249249249249

    # This is just one LLVM operation as opposed to `Int(x)`.
    return x % Int
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
        return rem(div(root_index, root_stride(cell_list, dim, 0)), n_roots[dim])
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

    # This is just one LLVM operation as opposed to `Int(x)`.
    return x % Int
end

# Inverse of `spread_bits_3d`. Extract every third bit of `x`.
@inline function compact_bits_3d(x)
    x = UInt64(x) & 0x1249249249249249
    x = (x | (x >> 2)) & 0x10c30c30c30c30c3
    x = (x | (x >> 4)) & 0x100f00f00f00f00f
    x = (x | (x >> 8)) & 0x001f0000ff0000ff
    x = (x | (x >> 16)) & 0x001f00000000ffff
    x = (x | (x >> 32)) & 0x00000000001fffff

    # This is just one LLVM operation as opposed to `Int(x)`.
    return x % Int
end

# Increment the coordinate that is stored in the bits `mask` of the Morton index `m`,
# where `m` must not have any non-zero bits outside of `mask`.
# For example, in 2D, the x-coordinate is stored in `mask = 0b0101` (lower 4 bits shown).
# Setting all bits outside of `mask` to one makes the carry of `+ 1` skip these bits.
# Then, removing the bits outside of `mask` yields the new Morton index.
# Incrementing the x-coordinate from 1 to 2 works as follows:
#
#     m                = 0b0001  (x-coordinate 1)
#     m | ~mask        = 0b1011
#     (m | ~mask) + 1  = 0b1100
#     ... & mask       = 0b0100  (x-coordinate 2)
#
# When all bits in `mask` are set (last cell in the root cell), the result wraps around
# to zero (first cell in the root cell).
@inline morton_increment(m, mask) = ((m | ~mask) + 1) & mask

# Decrement the coordinate that is stored in the bits `mask` of the Morton index `m`,
# where `m` must not have any bits outside of `mask` set.
# Since all bits outside of `mask` are zero, the borrow of `- 1` sets these bits on its way
# to the next bit in `mask`, and `& mask` clears them again.
# The first cell in the root cell wraps around to the last cell (all bits in `mask` set).
@inline morton_decrement(m, mask) = (m - 1) & mask

# The parts of the linear index (see `cell_index_part`) of the cells at level `level`
# with the coordinates `cell - 1`, `cell` and `cell + 1` in dimension `dim`.
#
# This is equivalent to calling `cell_index_part` for these three coordinates (with
# periodic wrap-around), but computes the Morton index only once and then increments and
# decrements it directly (see `morton_increment`).
# This significantly reduces the number of registers on GPUs.
@inline function neighbor_index_parts(cell_list, dim, cell, level, periodic_box)
    (; n_roots) = cell_list
    NDIMS = ndims(cell_list)

    # Coordinate of the root cell and coordinate within the root cell.
    root = cell >> level
    local_coord = cell - (root << level)
    last_local_coord = (1 << level) - 1

    # When the cell is the first or last cell in its root cell, the neighbor is in the
    # previous or next root cell.
    root_previous = ifelse(local_coord == 0,
                           previous_root(root, n_roots[dim], periodic_box), root)
    root_next = ifelse(local_coord == last_local_coord,
                       next_root(root, n_roots[dim], periodic_box), root)

    # When the neighbor is in the previous or next root cell, the Morton increment or
    # decrement wraps around to the correct local coordinate in that root cell.
    local_part = local_index_part(local_coord, dim, Val(NDIMS))
    mask = local_index_part(last_local_coord, dim, Val(NDIMS))
    local_previous = morton_decrement(local_part, mask)
    local_next = morton_increment(local_part, mask)

    stride = root_stride(cell_list, dim, level)

    return (root_previous * stride + local_previous,
            root * stride + local_part,
            root_next * stride + local_next)
end

# Without periodicity, the padding layer of root cells ensures that the neighboring root
# cells are inside the grid (see `root_grid`).
# With periodicity, the first and last root cell are neighbors.
@inline previous_root(root, n_roots, periodic_box::Nothing) = root - 1
@inline next_root(root, n_roots, periodic_box::Nothing) = root + 1
@inline function previous_root(root, n_roots, periodic_box)
    return ifelse(root == 0, n_roots - 1, root - 1)
end
@inline function next_root(root, n_roots, periodic_box)
    return ifelse(root == n_roots - 1, 0, root + 1)
end

# Zero-based Cartesian coordinates of the finest cell containing a query point with
# coordinates `coords`, which can be outside of the domain.
# With periodicity, `coords` must already be inside the periodic box.
# The cell is clamped to the grid, so that cells outside of the grid are treated like
# the empty cells in the padding layer of root cells.
@inline function query_finest_cell(coords, cell_list::TreeCellList)
    (; max_level) = cell_list

    cell = cell_coords(coords, cell_list, max_level)

    return clamp.(cell, 0, n_cells_per_dimension(cell_list, max_level) .- 1)
end

# The cell at level `level` around which the 3^d block of cells is searched in the query.
#
# Without periodicity, clamp to the grid without its outermost layer of cells.
# Since the domain is padded by a layer of root cells, this only affects points
# outside the domain.
@inline function query_cell(finest_cell, cell_list::TreeCellList, level,
                            periodic_box::Nothing)
    cell = finest_cell .>> (cell_list.max_level - level)

    return clamp.(cell, 1, n_cells_per_dimension(cell_list, level) .- 2)
end

@inline function query_cell(finest_cell, cell_list::TreeCellList, level, periodic_box)
    return finest_cell .>> (cell_list.max_level - level)
end

# Zero-based index of the finest cell containing the point with coordinates `coords`.
# Points must be inside the domain.
@inline function finest_cell_index(coords, cell_list::TreeCellList, periodic_box)
    (; max_level) = cell_list

    check_domain(coords, cell_list, periodic_box)

    # Clamp to the grid without the padding layer to avoid cells in the padding layer
    # due to rounding errors and for points exactly on the upper boundary of the domain.
    # With periodicity, clamp to the grid.
    padding = padding_cells(cell_list, max_level, periodic_box)
    cell = cell_coords(periodic_coords(coords, periodic_box), cell_list, max_level)
    cell_ = clamp.(cell, padding,
                   n_cells_per_dimension(cell_list, max_level) .- 1 .- padding)

    return cell_index(cell_list, cell_, max_level)
end

# Number of cells at level `level` in the padding layer of root cells
@inline padding_cells(cell_list, level, periodic_box::Nothing) = 1 << level
@inline padding_cells(cell_list, level, periodic_box) = 0

@inline function check_domain(coords, cell_list, periodic_box::Nothing)
    (; min_corner, max_corner) = cell_list

    # This also catches NaNs
    if !all(min_corner .<= coords .<= max_corner)
        error("particle coordinates are NaN or outside the domain bounds of the cell list")
    end
end

# With periodicity, points outside of the periodic box are moved into the periodic box
@inline function check_domain(coords, cell_list, periodic_box)
    if any(isnan, coords)
        error("particle coordinates are NaN")
    end
end

# Rebuild the tree from scratch for the points with IDs in `eachindex_points` with
# coordinates `coords`
# and radii `radii`.
function rebuild!(cell_list::TreeCellList, coords, radii, eachindex_points, periodic_box)
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
        finest_cell = finest_cell_index(point_coords, cell_list, periodic_box)
        i = pyramid_index(cell_list, finest_cell, max_level)
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
            split_cell!(cell_list, Tuple(cell) .- 1, level, periodic_box)
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
        finest_cell = finest_cell_index(point_coords, cell_list, periodic_box)
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

# The maximum search radius that is supported.
# Allow small tolerance to avoid errors due to rounding errors in the root cell size,
# for example when a periodic box of size 0.3 is divided into 3 root cells
# and the search radius is 0.1.
@inline function max_root_radius(cell_list)
    return minimum(cell_list.actual_root_cell_size) * (1 + 10 * eps(eltype(cell_list)))
end

function check_max_radius(cell_list)
    (; max_radius, max_radius_all, n_roots) = cell_list

    # The maximum radius of all points is the maximum of all root cells
    max_radius_all_ = maximum(view(max_radius, 1:prod(n_roots)),
                              init = zero(eltype(max_radius)))
    fill!(max_radius_all, max_radius_all_)

    # The 3^d block of root cells around a root cell only contains all neighbors
    # when all radii are smaller than the root cells
    if max_radius_all_ > max_root_radius(cell_list)
        error("the search radii must not exceed the `root_cell_size` of the cell list")
    end
end

# Decide whether the cell with zero-based Cartesian coordinates `cell` at level `level`
# is split, and store the level of the leaf containing this cell in `leaf_level`.
# Note that this requires the parent level to be processed already.
@inline function split_cell!(cell_list, cell, level, periodic_box)
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
    # Empty cells are never split, so the cells in the padding layer of root cells
    # are leaves at level 0.
    if level < max_level && n_points[i] > capacity &&
       can_split(cell_list, cell, level, periodic_box)
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
#
# Only non-empty cells are split, which are not in the padding layer of root cells.
# Therefore, the 3^d block around this cell is inside the grid.
# With periodicity, the 3^d block contains the periodic images of the cells
# on the other side of the periodic box.
@inline function can_split(cell_list, cell, level, periodic_box)
    (; max_radius) = cell_list
    NDIMS = ndims(cell_list)
    n_cells = n_cells_per_dimension(cell_list, level)

    # The cells are not necessarily cubic with periodicity. Points outside of the 3^d
    # block are at least as far away as the smallest cell size.
    child_size = minimum(cell_size(cell_list, level + 1))

    for neighbor_cell_ in CartesianIndices(ntuple(i -> (cell[i] - 1):(cell[i] + 1),
                                Val(NDIMS)))
        neighbor_cell = periodic_cell(Tuple(neighbor_cell_), n_cells, periodic_box)

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
