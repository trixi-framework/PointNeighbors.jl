@doc raw"""
    TreeCellList(; min_corner, max_corner, max_level,
                 root_cell_size = maximum(max_corner .- min_corner),
                 backend = DynamicVectorOfVectors{Int32}, max_points_per_cell = 100)

Cell list for the [`TreeNeighborhoodSearch`](@ref), storing points in the cells of
a quadtree (2D) or octree (3D).

The domain is covered by a grid of cubic root cells of size `root_cell_size`.
By default, a single root cell covers the whole domain.
Each root cell is the root of a tree, in which a cell at level ``L``
has the size `root_cell_size` ``/ 2^L``.
The finest possible level is `max_level`.
The cells on each level are sorted by root cell first and then in Morton order inside
each root cell, so that all descendants of a cell are stored contiguously.

With a uniform search radius, all leaves of the tree are at the same level, which is the
finest level with cells that are at least as large as the search radius,
but at most `max_level`. This level is chosen by the [`TreeNeighborhoodSearch`](@ref).

!!! warning "Experimental implementation"
    This is an experimental feature and may change in any future releases.

# Keywords
- `min_corner`: Coordinates of the domain corner in negative coordinate directions.
- `max_corner`: Coordinates of the domain corner in positive coordinate directions.
                All points must be inside the domain.
- `max_level`:  The finest level of the trees.
- `root_cell_size = maximum(max_corner .- min_corner)`: Size of the root cells.
                For elongated domains, a smaller size can be used to obtain a grid of root
                cells and avoid wasting memory for cells outside the domain.
                The search radius must not exceed `root_cell_size`.
                For the best performance, use a power of two times the search radius,
                so that the leaves have exactly the size of the search radius.
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
    n_roots        :: NTuple{NDIMS, Int}
    max_level      :: Int
    # Level of all leaves, chosen by the neighborhood search in `allocate_leaves`
    leaf_level::Int
    # One list of points for each leaf, sorted by `cell_index`
    cells::C
end

function TreeCellList(; min_corner, max_corner, max_level,
                      root_cell_size = maximum(max_corner .- min_corner),
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

    # The lists for the leaves are allocated by `allocate_leaves`, which is called
    # in the constructor of the `TreeNeighborhoodSearch`.
    cells = construct_backend(backend, 0, max_points_per_cell)

    return TreeCellList(min_corner_, max_corner_, root_cell_size_, n_roots, max_level,
                        max_level, cells)
end

# Return a copy of `cell_list` with the leaves at the finest level with cells that are
# at least as large as `search_radius`, and with one list for each leaf
function allocate_leaves(cell_list::TreeCellList, search_radius)
    (; min_corner, max_corner, root_cell_size, n_roots, max_level, cells) = cell_list
    NDIMS = ndims(cell_list)

    # The 3^d block of cells around a leaf only contains all neighbors if the leaves
    # are at least as large as the search radius
    if search_radius > root_cell_size
        throw(ArgumentError("the search radius must not exceed the `root_cell_size` of " *
                            "the `TreeCellList`"))
    end

    leaf_level = 0
    while leaf_level < max_level && cell_size(cell_list, leaf_level + 1) >= search_radius
        leaf_level += 1
    end

    n_leaves = prod(n_roots) << (NDIMS * leaf_level)
    cells_ = construct_backend(typeof(cells), n_leaves, max_inner_length(cells, 100))

    return TreeCellList(min_corner, max_corner, root_cell_size, n_roots, max_level,
                        leaf_level, cells_)
end

@inline Base.ndims(::TreeCellList{NDIMS}) where {NDIMS} = NDIMS
@inline Base.eltype(::TreeCellList{<:Any, ELTYPE}) where {ELTYPE} = ELTYPE

function supported_update_strategies(::TreeCellList)
    return (SerialUpdate,)
end

function copy_cell_list(cell_list::TreeCellList)
    (; min_corner, max_corner, max_level, root_cell_size, cells) = cell_list

    return TreeCellList(; min_corner, max_corner, max_level, root_cell_size,
                        backend = typeof(cells),
                        max_points_per_cell = max_inner_length(cells, 100))
end

# Size of the cells at level `level`
@inline function cell_size(cell_list::TreeCellList, level)
    (; root_cell_size) = cell_list

    return root_cell_size / (1 << level)
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

# Zero-based index of the leaf containing the point with coordinates `coords`.
# Points must be inside the domain.
@inline function leaf_index(coords, cell_list::TreeCellList)
    (; min_corner, max_corner, leaf_level) = cell_list

    # This also catches NaNs
    if !all(min_corner .<= coords .<= max_corner)
        error("particle coordinates are NaN or outside the domain bounds of the cell list")
    end

    # Clamp to the grid to avoid out of bounds cells due to rounding errors
    # and for points exactly on the upper boundary of the grid.
    cell = cell_coords(coords, cell_list, leaf_level)
    cell_ = clamp.(cell, 0, n_cells_per_dimension(cell_list, leaf_level) .- 1)

    return cell_index(cell_list, cell_, leaf_level)
end

# Rebuild the cell list from scratch for the points `eachindex_points`
# with coordinates `coords`
function rebuild!(cell_list::TreeCellList, coords, eachindex_points)
    (; cells) = cell_list
    NDIMS = ndims(cell_list)

    for i in eachindex(cells)
        emptyat!(cells, i)
    end

    # Add each point to its leaf
    for point in eachindex_points
        point_coords = extract_svector(coords, Val(NDIMS), point)
        pushat!(cells, leaf_index(point_coords, cell_list) + 1, point)
    end

    return cell_list
end

# The points of the leaf with zero-based index `leaf_index`
@propagate_inbounds function points_in_leaf(cell_list::TreeCellList, leaf_index)
    return cell_list.cells[leaf_index + 1]
end
