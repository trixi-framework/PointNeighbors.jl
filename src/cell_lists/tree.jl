@doc raw"""
    TreeCellList(; min_corner, max_corner, root_cell_size = zero(eltype(min_corner)),
                 backend = DynamicVectorOfVectors{Int32}, max_points_per_cell = 100)

Cell list for the [`TreeNeighborhoodSearch`](@ref), storing points in the cells of
a quadtree (2D) or octree (3D).

The domain is covered by a grid of cubic root cells of size `root_cell_size`.
With periodicity, the root cells tile the periodic box exactly, so they can be slightly
larger than `root_cell_size` and differ in size between the dimensions.
Each root cell is the root of a tree, in which a cell at level ``L``
has the size of the root cell ``/ 2^L``.
The cells on each level are sorted by root cell first and then in Morton order inside
each root cell, so that all descendants of a cell are stored contiguously.

With a uniform search radius, all leaves of the tree are at the same level, which is the
finest level with cells that are at least as large as the search radius.
This level is chosen by the [`TreeNeighborhoodSearch`](@ref).

Only set `min_corner` and `max_corner` and use the default values for the other arguments
to create an empty "template" cell list that can be used to create an empty "template"
neighborhood search. When copying the template neighborhood search with
[`copy_neighborhood_search`](@ref), the size of the root cells is the search radius.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in any future releases.

# Keywords
- `min_corner`: Coordinates of the domain corner in negative coordinate directions.
- `max_corner`: Coordinates of the domain corner in positive coordinate directions.
                All points must be inside the domain.
- `root_cell_size`: Size of the root cells, which must be at least as large as the
                search radius. Root cells of the size of the search radius are the most
                efficient choice. Then, the root cells are the leaves of the tree.
                Use the default of zero to create a template (see above).
- `backend = DynamicVectorOfVectors{Int32}`: Type of the data structure to store the
    points of each leaf. Can be
    - `Vector{Vector{Int32}}`: Scattered memory, but very memory-efficient.
    - `DynamicVectorOfVectors{Int32}`: Contiguous memory, optimizing cache-hits.
- `max_points_per_cell = 100`: Maximum number of points per leaf. This will be used to
                allocate the `DynamicVectorOfVectors`. It is not used with
                the `Vector{Vector{Int32}}` backend.
"""
struct TreeCellList{NDIMS, ELTYPE, C} <: AbstractCellList
    min_corner     :: SVector{NDIMS, ELTYPE}
    max_corner     :: SVector{NDIMS, ELTYPE}
    # Size of the root cells requested by the user
    root_cell_size :: ELTYPE
    # The grid of root cells, see `root_grid`
    grid_min_corner :: SVector{NDIMS, ELTYPE}
    # Actual size of the root cells in each dimension. Without periodicity, this is
    # `root_cell_size` in every dimension. With periodicity, it can be larger.
    actual_root_cell_size :: SVector{NDIMS, ELTYPE}
    n_roots               :: NTuple{NDIMS, Int}
    # Level of all leaves, chosen by the neighborhood search in `build_tree`
    leaf_level::Int
    # One list of points for each leaf, sorted by `cell_index`
    cells::C
end

function TreeCellList(; min_corner, max_corner,
                      root_cell_size = zero(eltype(min_corner)),
                      backend = DynamicVectorOfVectors{Int32}, max_points_per_cell = 100)
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

    ELTYPE = promote_type(eltype(min_corner), eltype(max_corner), typeof(root_cell_size))
    min_corner_ = SVector{NDIMS, ELTYPE}(Tuple(min_corner))
    max_corner_ = SVector{NDIMS, ELTYPE}(Tuple(max_corner))
    root_cell_size_ = convert(ELTYPE, root_cell_size)

    (grid_min_corner, actual_root_cell_size,
     n_roots) = root_grid(min_corner_, max_corner_, root_cell_size_, nothing)

    # The lists for the leaves are allocated by `build_tree`, which is called
    # in the constructor of the `TreeNeighborhoodSearch`.
    cells = construct_backend(backend, 0, max_points_per_cell)

    return TreeCellList(min_corner_, max_corner_, root_cell_size_, grid_min_corner,
                        actual_root_cell_size, n_roots, 0, cells)
end

# The grid of root cells as `(grid_min_corner, actual_root_cell_size, n_roots)`, where
# `grid_min_corner` is the corner of the grid in negative coordinate directions
# and `actual_root_cell_size` is the actual size of the root cells in each dimension.
#
# Without periodicity, the grid has one padding layer of root cells around the domain.
# Then, the 3^d block of leaves around each leaf inside the domain is inside the grid,
# and we don't need to clip this block to the grid in the query.
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

# With periodicity, the root cells tile the periodic box exactly. They are therefore
# slightly larger than `root_cell_size` when the periodic box is not a multiple of
# `root_cell_size`. In the query, the neighboring leaves of the leaves at the boundary
# are the leaves at the opposite boundary.
function root_grid(min_corner, max_corner, root_cell_size, periodic_box)
    if !(min_corner ≈ periodic_box.min_corner && max_corner ≈ periodic_box.max_corner)
        throw(ArgumentError("the domain of the `TreeCellList` must be the periodic box"))
    end

    # Allow small tolerance to avoid inefficient larger cells due to machine
    # rounding errors, as in the `GridNeighborhoodSearch`
    n_roots = Tuple(floor.(Int, (periodic_box.size .+ 10eps()) ./ root_cell_size))

    if any(<(1), n_roots)
        throw(ArgumentError("the `root_cell_size` must not exceed the size of the " *
                            "periodic box"))
    end

    # Convert to the element type of the cell list, which can differ from the element type
    # of the periodic box.
    ELTYPE = eltype(min_corner)
    grid_min_corner = convert.(ELTYPE, periodic_box.min_corner)
    actual_root_cell_size = convert.(ELTYPE, periodic_box.size ./ n_roots)

    return grid_min_corner, actual_root_cell_size, n_roots
end

# Return a copy of `cell_list` with the grid of root cells for `periodic_box`,
# the leaves at the finest level with cells that are at least as large as `search_radius`,
# and one list for each leaf
function build_tree(cell_list::TreeCellList, search_radius, periodic_box)
    (; min_corner, max_corner, root_cell_size, cells) = cell_list
    NDIMS = ndims(cell_list)

    # This also catches NaNs, as opposed to `search_radius > 0`.
    if !(search_radius >= 0)
        throw(ArgumentError("the search radius must not be negative or NaN"))
    end

    if iszero(search_radius)
        # This is an empty "template" neighborhood search to be used with
        # `copy_neighborhood_search`
        return cell_list
    end

    if iszero(root_cell_size)
        throw(ArgumentError("the `root_cell_size` of the `TreeCellList` must be set, " *
                            "unless creating a template neighborhood search"))
    end

    # The 3^d block of leaves around a leaf only contains all neighbors if the leaves
    # are at least as large as the search radius
    if search_radius > root_cell_size
        throw(ArgumentError("the search radius must not exceed the `root_cell_size` of " *
                            "the `TreeCellList`"))
    end

    (grid_min_corner, actual_root_cell_size,
     n_roots) = root_grid(min_corner, max_corner, root_cell_size, periodic_box)

    # The Morton index of a leaf inside a root cell must fit into an `Int64`.
    # Larger leaves are less efficient, but still contain all neighbors.
    max_leaf_level = div(60, NDIMS)
    leaf_level = 0
    while leaf_level < max_leaf_level &&
        minimum(actual_root_cell_size) / (1 << (leaf_level + 1)) >= search_radius
        leaf_level += 1
    end

    # With periodicity, the 3^d block of leaves around a leaf must not contain multiple
    # periodic images of the same leaf
    if !isnothing(periodic_box) && any(n -> (n << leaf_level) < 3, n_roots)
        throw(ArgumentError("the `TreeNeighborhoodSearch` needs at least 3 leaves " *
                            "in each dimension when used with periodicity. " *
                            "For very small problems, use the " *
                            "`TrivialNeighborhoodSearch` instead."))
    end

    n_leaves = prod(n_roots) << (NDIMS * leaf_level)
    cells_ = construct_backend(typeof(cells), n_leaves, max_inner_length(cells, 100))

    return TreeCellList(min_corner, max_corner, root_cell_size, grid_min_corner,
                        actual_root_cell_size, n_roots, leaf_level, cells_)
end

@inline Base.ndims(::TreeCellList{NDIMS}) where {NDIMS} = NDIMS
@inline Base.eltype(::TreeCellList{<:Any, ELTYPE}) where {ELTYPE} = ELTYPE

function supported_update_strategies(::TreeCellList)
    return (SerialUpdate,)
end

# Return an empty cell list with the same domain as `cell_list` and root cells of the size
# of the search radius `search_radius`.
function copy_cell_list(cell_list::TreeCellList, search_radius)
    (; min_corner, max_corner, cells) = cell_list

    return TreeCellList(; min_corner, max_corner, root_cell_size = search_radius,
                        backend = typeof(cells),
                        max_points_per_cell = max_inner_length(cells, 100))
end

# Size of the leaves in each dimension
@inline function leaf_size(cell_list::TreeCellList)
    (; actual_root_cell_size, leaf_level) = cell_list

    return actual_root_cell_size ./ (1 << leaf_level)
end

# Number of cells at level `level` in each dimension
@inline function n_cells_per_dimension(cell_list::TreeCellList, level)
    return cell_list.n_roots .<< level
end

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
@inline function cell_index(cell_list::TreeCellList, cell, level)
    NDIMS = ndims(cell_list)

    return sum(ntuple(dim -> cell_index_part(cell_list, dim, cell[dim], level),
                      Val(NDIMS)))
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

# The parts of the linear index (see `cell_index_part`) of the leaves with the coordinates
# `leaf - 1`, `leaf` and `leaf + 1` in dimension `dim`.
#
# This is equivalent to calling `cell_index_part` for these three coordinates (with
# periodic wrap-around), but computes the Morton index only once and then increments and
# decrements it directly (see `morton_increment`).
# This significantly reduces the number of registers on GPUs.
@inline function neighbor_index_parts(cell_list, dim, leaf, periodic_box)
    (; n_roots, leaf_level) = cell_list
    NDIMS = ndims(cell_list)

    # Coordinate of the root cell and coordinate within the root cell.
    root = leaf >> leaf_level
    local_coord = leaf - (root << leaf_level)
    last_local_coord = (1 << leaf_level) - 1

    # When the leaf is the first or last leaf in its root cell, the neighbor is in the
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

    stride = root_stride(cell_list, dim, leaf_level)

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

# Zero-based Cartesian coordinates of the leaf containing the point with coordinates
# `coords`, which can be outside of the domain.
# With periodicity, `coords` must already be inside the periodic box.
@inline function leaf_coords(coords, cell_list::TreeCellList, periodic_box)
    (; grid_min_corner) = cell_list

    leaf = Tuple(floor_to_int.((coords .- grid_min_corner) ./ leaf_size(cell_list)))

    return clamp_leaf(leaf, n_cells_per_dimension(cell_list, cell_list.leaf_level),
                      periodic_box)
end

# Without periodicity, clamp to the grid without its outermost layer of leaves.
# Since the domain is padded by a layer of root cells, this only affects points
# outside the domain.
@inline clamp_leaf(leaf, n_leaves, periodic_box::Nothing) = clamp.(leaf, 1, n_leaves .- 2)

# With periodicity, clamp to the grid to avoid out of bounds leaves due to rounding errors
# for points on the upper boundary of the periodic box
@inline clamp_leaf(leaf, n_leaves, periodic_box) = clamp.(leaf, 0, n_leaves .- 1)

# Zero-based index of the leaf containing the point with coordinates `coords`.
# Points must be inside the domain.
@inline function leaf_index(coords, cell_list::TreeCellList, periodic_box)
    check_domain(coords, cell_list, periodic_box)

    leaf = leaf_coords(periodic_coords(coords, periodic_box), cell_list, periodic_box)

    return cell_index(cell_list, leaf, cell_list.leaf_level)
end

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

# Rebuild the cell list from scratch for the points with IDs in `eachindex_points`
# with coordinates `coords`.
function rebuild!(cell_list::TreeCellList, coords, eachindex_points, periodic_box)
    (; cells) = cell_list
    NDIMS = ndims(cell_list)

    for i in eachindex(cells)
        emptyat!(cells, i)
    end

    # Add each point to its leaf.
    for point in eachindex_points
        point_coords = extract_svector(coords, Val(NDIMS), point)
        pushat!(cells, leaf_index(point_coords, cell_list, periodic_box) + 1, point)
    end

    return cell_list
end

# The points of the leaf with zero-based index `leaf_index`.
@propagate_inbounds function points_in_leaf(cell_list::TreeCellList, leaf_index)
    return cell_list.cells[leaf_index + 1]
end
