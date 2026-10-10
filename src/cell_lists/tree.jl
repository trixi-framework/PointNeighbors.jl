@doc raw"""
    TreeCellList(; min_corner, max_corner, max_level,
                 root_cell_size = maximum(max_corner .- min_corner), capacity = 0)

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

Internally, the points are sorted by the cells at level `max_level` in Morton order.
This way, the points of each cell at each level are stored contiguously in memory.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in any future releases.

!!! note "Memory usage"
    The tree data structures are stored densely for all levels. They therefore require
    memory proportional to the number of cells at level `max_level`, which is
    (number of root cells) ``\times 2^{d \cdot \text{max\_level}}``.

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
- `capacity = 0`: Only split cells that contain more than `capacity` points.
"""
struct TreeCellList{NDIMS, ELTYPE, MR, MRA, NP, LL, CO, P, PC} <: AbstractCellList
    min_corner     :: SVector{NDIMS, ELTYPE}
    max_corner     :: SVector{NDIMS, ELTYPE}
    root_cell_size :: ELTYPE
    n_roots        :: NTuple{NDIMS, Int}
    max_level      :: Int
    capacity       :: Int
    # Pyramids with one entry for each cell at each level, see `pyramid_index`
    max_radius     :: MR  # Maximum radius of all points in this cell
    max_radius_all :: MRA # Maximum radius of all points (one-element vector)
    n_points       :: NP  # Number of points in this cell
    leaf_level     :: LL  # Level of the leaf containing this cell or -1 for split cells
    # Compressed storage of the points sorted by the cells at the finest level
    cell_offsets :: CO  # Points of finest cell `c` are `points[cell_offsets[c] + 1:cell_offsets[c + 1]]`
    points       :: P
    point_cells  :: PC  # Finest cell of each point
end

function TreeCellList(; min_corner, max_corner, max_level,
                      root_cell_size = maximum(max_corner .- min_corner), capacity = 0)
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

    max_radius = zeros(ELTYPE, n_cells)
    max_radius_all = zeros(ELTYPE, 1)
    n_points = zeros(Int32, n_cells)
    leaf_level = zeros(Int8, n_cells)
    cell_offsets = zeros(Int32, n_finest_cells + 1)
    points = Int32[]
    point_cells = Int[]

    return TreeCellList(min_corner_, max_corner_, root_cell_size_, n_roots, max_level,
                        capacity, max_radius, max_radius_all, n_points, leaf_level,
                        cell_offsets, points, point_cells)
end

@inline Base.ndims(::TreeCellList{NDIMS}) where {NDIMS} = NDIMS
@inline Base.eltype(::TreeCellList{<:Any, ELTYPE}) where {ELTYPE} = ELTYPE

function supported_update_strategies(::TreeCellList)
    return (SerialUpdate,)
end

function copy_cell_list(cell_list::TreeCellList)
    (; min_corner, max_corner, max_level, root_cell_size, capacity) = cell_list

    return TreeCellList(; min_corner, max_corner, max_level, root_cell_size, capacity)
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
    (; n_roots) = cell_list
    NDIMS = ndims(cell_list)

    root = cell .>> level
    local_cell = cell .- (root .<< level)

    # Column-major linear index of the root cell
    root_index = root[NDIMS]
    for dim in (NDIMS - 1):-1:1
        root_index = root_index * n_roots[dim] + root[dim]
    end

    return (root_index << (NDIMS * level)) | morton_index(local_cell)
end

# Interleave the bits of the coordinates to obtain the Morton index (Z-order).
@inline morton_index(cell::NTuple{1}) = cell[1]

@inline function morton_index(cell::NTuple{2})
    return spread_bits_2d(cell[1]) | (spread_bits_2d(cell[2]) << 1)
end

@inline function morton_index(cell::NTuple{3})
    return spread_bits_3d(cell[1]) | (spread_bits_3d(cell[2]) << 1) |
           (spread_bits_3d(cell[3]) << 2)
end

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
    (; max_level, max_radius, n_points, points, point_cells) = cell_list
    NDIMS = ndims(cell_list)

    resize!(points, length(eachindex_points))
    resize!(point_cells, size(coords, 2))

    # Only the finest level has to be reset. The coarser levels are computed from it below.
    finest_cells = (level_offset(cell_list, max_level) + 1):length(n_points)
    view(max_radius, finest_cells) .= 0
    view(n_points, finest_cells) .= 0

    # Count points and compute the maximum radius of each finest cell
    for point in eachindex_points
        point_coords = extract_svector(coords, Val(NDIMS), point)
        cell = finest_cell_index(point_coords, cell_list)
        point_cells[point] = cell

        i = pyramid_index(cell_list, cell, max_level)
        n_points[i] += 1
        max_radius[i] = max(max_radius[i], radii[point])
    end

    # Compute the counts and maximum radii of all coarser cells from their children
    for level in (max_level - 1):-1:0
        for cell in
            0:(level_offset(cell_list, level + 1) - level_offset(cell_list, level) - 1)

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

    sort_points!(cell_list, eachindex_points)

    return cell_list
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
    max_radius_all[1] = maximum(view(max_radius, 1:prod(n_roots)), init = 0)

    # With a single root cell, the 3^d block around the root cell contains all points.
    # With multiple root cells, we can only guarantee to find all neighbors when
    # all radii are smaller than the root cells.
    if prod(n_roots) > 1 && max_radius_all[1] > root_cell_size
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
# `distance <= (h_i + h_j) / 2` are in the 3^d block around this leaf at level `L`.
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

# Sort the points by their finest cells (counting sort)
function sort_points!(cell_list, eachindex_points)
    (; max_level, n_points, cell_offsets, points, point_cells) = cell_list

    # Exclusive prefix sum of the number of points in all finest cells
    offset = level_offset(cell_list, max_level)
    cell_offsets[1] = 0
    for cell in 1:(length(cell_offsets) - 1)
        cell_offsets[cell + 1] = cell_offsets[cell] + n_points[offset + cell]
    end

    # Put each point in its place. We are using the counts of the finest cells as counters
    # here, which then point to the next free position in the cell, counting backwards.
    for point in eachindex_points
        cell = point_cells[point]
        i = offset + cell + 1
        points[cell_offsets[cell + 1] + n_points[i]] = point
        n_points[i] -= 1
    end

    # Restore the counts
    for cell in 1:(length(cell_offsets) - 1)
        n_points[offset + cell] = cell_offsets[cell + 1] - cell_offsets[cell]
    end

    return cell_list
end

# Level of the leaf containing the finest cell with zero-based index `finest_cell`
@propagate_inbounds function leaf_level(cell_list::TreeCellList, finest_cell)
    (; leaf_level, max_level) = cell_list

    return Int(leaf_level[pyramid_index(cell_list, finest_cell, max_level)])
end

# The points in the finest cells `first_cell:(last_cell - 1)` (zero-based indices).
# Because the points are sorted by their finest cells, this is a contiguous range.
@propagate_inbounds function points_in_finest_cells(cell_list::TreeCellList, first_cell,
                                                    last_cell)
    (; cell_offsets, points) = cell_list

    return view(points, (cell_offsets[first_cell + 1] + 1):cell_offsets[last_cell + 1])
end
