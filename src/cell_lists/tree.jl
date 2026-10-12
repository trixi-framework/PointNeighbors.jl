@doc raw"""
    TreeCellList(; min_corner, max_corner, root_cell_size = zero(eltype(min_corner)),
                 backend = DynamicVectorOfVectors{Int32}, max_points_per_cell = 100)

Cell list for the [`TreeNeighborhoodSearch`](@ref), storing points in the cells of
a quadtree (2D) or octree (3D).

The domain is covered by a grid of cubic root cells of size `root_cell_size`.
Each root cell is the root of a tree, in which a cell at level ``L``
has the size `root_cell_size` ``/ 2^L``.
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
    root_cell_size :: ELTYPE
    # The grid of root cells, see `root_grid`
    grid_min_corner :: SVector{NDIMS, ELTYPE}
    root_size       :: SVector{NDIMS, ELTYPE}
    n_roots         :: NTuple{NDIMS, Int}
    # Level of all leaves, chosen by the neighborhood search in `allocate_leaves`
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

    if root_cell_size < 0
        throw(ArgumentError("`root_cell_size` must not be negative"))
    end

    ELTYPE = promote_type(eltype(min_corner), eltype(max_corner), typeof(root_cell_size))
    min_corner_ = SVector{NDIMS, ELTYPE}(Tuple(min_corner))
    max_corner_ = SVector{NDIMS, ELTYPE}(Tuple(max_corner))
    root_cell_size_ = convert(ELTYPE, root_cell_size)

    grid_min_corner, root_size,
    n_roots = root_grid(min_corner_, max_corner_,
                        root_cell_size_, nothing)

    # The lists for the leaves are allocated by `allocate_leaves`, which is called
    # in the constructor of the `TreeNeighborhoodSearch`.
    cells = construct_backend(backend, 0, max_points_per_cell)

    return TreeCellList(min_corner_, max_corner_, root_cell_size_, grid_min_corner,
                        root_size, n_roots, 0, cells)
end

# The grid of root cells as `(grid_min_corner, root_size, n_roots)`, where
# `grid_min_corner` is the corner of the grid in negative coordinate directions
# and `root_size` is the size of the root cells in each dimension.
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
    root_size = root_cell_size .+ zero(min_corner)

    return grid_min_corner, root_size, n_roots
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

    root_size = periodic_box.size ./ n_roots

    return periodic_box.min_corner, root_size, n_roots
end

# Return a copy of `cell_list` with the grid of root cells for `periodic_box`,
# the leaves at the finest level with cells that are at least as large as `search_radius`,
# and one list for each leaf
function allocate_leaves(cell_list::TreeCellList, search_radius, periodic_box)
    (; min_corner, max_corner, root_cell_size, cells) = cell_list
    NDIMS = ndims(cell_list)

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

    grid_min_corner, root_size,
    n_roots = root_grid(min_corner, max_corner,
                        root_cell_size, periodic_box)

    # The Morton index of a leaf inside a root cell must fit into an `Int64`.
    # Larger leaves are less efficient, but still contain all neighbors.
    max_leaf_level = div(60, NDIMS)
    leaf_level = 0
    while leaf_level < max_leaf_level &&
        minimum(root_size) / (1 << (leaf_level + 1)) >= search_radius
        leaf_level += 1
    end

    # With periodicity, the 3^d block of leaves around a leaf must not contain multiple
    # periodic images of the same leaf
    if !isnothing(periodic_box) && any(n -> (n << leaf_level) < 3, n_roots)
        throw(ArgumentError("the `TreeNeighborhoodSearch` needs at least 3 leaves " *
                            "in each dimension when used with periodicity. " *
                            "Please use no NHS for very small problems."))
    end

    n_leaves = prod(n_roots) << (NDIMS * leaf_level)
    cells_ = construct_backend(typeof(cells), n_leaves, max_inner_length(cells, 100))

    return TreeCellList(min_corner, max_corner, root_cell_size, grid_min_corner,
                        root_size, n_roots, leaf_level, cells_)
end

@inline Base.ndims(::TreeCellList{NDIMS}) where {NDIMS} = NDIMS
@inline Base.eltype(::TreeCellList{<:Any, ELTYPE}) where {ELTYPE} = ELTYPE

function supported_update_strategies(::TreeCellList)
    return (SerialUpdate,)
end

# Return an empty cell list with the same domain as `cell_list` and root cells of the size
# of the search radius `search_radius`
function copy_cell_list(cell_list::TreeCellList, search_radius)
    (; min_corner, max_corner, cells) = cell_list

    return TreeCellList(; min_corner, max_corner, root_cell_size = search_radius,
                        backend = typeof(cells),
                        max_points_per_cell = max_inner_length(cells, 100))
end

# Size of the leaves in each dimension
@inline function leaf_size(cell_list::TreeCellList)
    (; root_size, leaf_level) = cell_list

    return root_size ./ (1 << leaf_level)
end

# Number of cells at level `level` in each dimension
@inline function n_cells_per_dimension(cell_list::TreeCellList, level)
    return cell_list.n_roots .<< level
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
# in this dimension. See `neighbor_index_parts` for how this is used in the query.
@inline function cell_index_part(cell_list::TreeCellList, dim, coord, level)
    # Coordinate of the root cell and coordinate within the root cell
    root = coord >> level
    local_coord = coord - (root << level)

    return root_index_part(cell_list, dim, root, level) |
           (spread_bits(local_coord, Val(ndims(cell_list))) << (dim - 1))
end

# The part of the linear cell index at level `level` of the root cell with coordinate
# `root` in dimension `dim`. The root cells are sorted in column-major order.
@inline function root_index_part(cell_list, dim, root, level)
    (; n_roots) = cell_list

    stride = 1
    for i in 1:(dim - 1)
        stride *= n_roots[i]
    end

    return (root * stride) << (ndims(cell_list) * level)
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

# All bits of the Morton index of a cell inside a root cell that belong to the first
# dimension. The bits of dimension `dim` are obtained by shifting by `dim - 1`.
@inline morton_mask(::Val{1}) = -1
@inline morton_mask(::Val{2}) = reinterpret(Int, 0x5555555555555555)
@inline morton_mask(::Val{3}) = reinterpret(Int, 0x1249249249249249)

# The parts of the linear index (see `cell_index_part`) of the leaves with the coordinates
# `leaf - 1`, `leaf` and `leaf + 1` in dimension `dim`.
#
# Instead of computing each part with `cell_index_part`, we compute the parts of the
# neighboring leaves by incrementing and decrementing the Morton index directly.
# For example, in 2D, the bits of the x-coordinate are at the even bit positions of the
# Morton index. Setting all odd bits to one and adding one increments the x-coordinate,
# since the carry is propagated through the odd bits.
#
# When the leaf `leaf - 1` or `leaf + 1` is in the neighboring root cell, this Morton
# increment or decrement wraps around to the correct coordinate inside the neighboring root
# cell, and we only have to move the root part to the neighboring root cell.
@inline function neighbor_index_parts(cell_list, dim, leaf, periodic_box)
    (; n_roots, leaf_level) = cell_list
    NDIMS = ndims(cell_list)

    # Coordinate of the root cell and part of the Morton index inside the root cell
    root = leaf >> leaf_level
    local_part = spread_bits(leaf - (root << leaf_level), Val(NDIMS)) << (dim - 1)

    # The part of the root cell and the difference to the part of the next root cell
    root_part = root_index_part(cell_list, dim, root, leaf_level)
    root_step = root_index_part(cell_list, dim, 1, leaf_level)

    # The bits of the Morton index inside the root cell that belong to dimension `dim`
    mask = (morton_mask(Val(NDIMS)) << (dim - 1)) & ((1 << (NDIMS * leaf_level)) - 1)
    step = 1 << (dim - 1)

    # Morton increment and decrement, which are zero or `mask` (all bits of this dimension
    # set) when leaving the root cell
    local_next = ((local_part | ~mask) + step) & mask
    local_previous = (local_part - step) & mask

    # Move the root part to the neighboring root cell when leaving the root cell.
    # Note that this is cheaper than computing the root parts of the neighboring root cells
    # with `root_index_part`, which significantly reduces the number of registers on GPUs.
    root_next = ifelse(local_next == 0,
                       root_part + next_root_step(root, root_step, n_roots[dim],
                                      periodic_box),
                       root_part)
    root_previous = ifelse(local_part == 0,
                           root_part - previous_root_step(root, root_step, n_roots[dim],
                                              periodic_box),
                           root_part)

    return root_previous + local_previous, root_part + local_part, root_next + local_next
end

# The difference between the root parts of the next root cell and the root cell `root`
# in one dimension, where `root_step` is the difference between neighboring root cells.
# With periodicity, the next root cell of the last root cell is the first root cell.
@inline next_root_step(root, root_step, n_roots, periodic_box::Nothing) = root_step
@inline function next_root_step(root, root_step, n_roots, periodic_box)
    return ifelse(root == n_roots - 1, -(n_roots - 1) * root_step, root_step)
end

# The difference between the root parts of the root cell `root` and the previous root cell.
# With periodicity, the previous root cell of the first root cell is the last root cell.
@inline previous_root_step(root, root_step, n_roots, periodic_box::Nothing) = root_step
@inline function previous_root_step(root, root_step, n_roots, periodic_box)
    return ifelse(root == 0, -(n_roots - 1) * root_step, root_step)
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

# Without periodicity, clamp to the grid without the outermost layer of leaves, so that the
# 3^d block of leaves around the leaf is inside the grid.
# Since the outermost layer is in the padding layer of root cells, this only affects points
# outside of the domain and points moved there by rounding errors. The block around the
# clamped leaf still contains all neighbors of these points, since points that are more than
# one leaf away from the domain don't have any neighbors.
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

# Rebuild the cell list from scratch for the points `eachindex_points`
# with coordinates `coords`
function rebuild!(cell_list::TreeCellList, coords, eachindex_points, periodic_box)
    (; cells) = cell_list
    NDIMS = ndims(cell_list)

    for i in eachindex(cells)
        emptyat!(cells, i)
    end

    # Add each point to its leaf
    for point in eachindex_points
        point_coords = extract_svector(coords, Val(NDIMS), point)
        pushat!(cells, leaf_index(point_coords, cell_list, periodic_box) + 1, point)
    end

    return cell_list
end

# The points of the leaf with zero-based index `leaf_index`
@propagate_inbounds function points_in_leaf(cell_list::TreeCellList, leaf_index)
    return cell_list.cells[leaf_index + 1]
end
