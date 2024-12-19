struct FaceNeighborhoodSearch{NDIMS, CL, ELTYPE} <: AbstractNeighborhoodSearch
    cell_list     :: CL
    search_radius :: ELTYPE
    periodic_box  :: Nothing
    n_cells       :: NTuple{NDIMS, Int}
    cell_size     :: NTuple{NDIMS, ELTYPE} # Required to calculate cell index
end

function FaceNeighborhoodSearch{NDIMS}(; cell_list = DictionaryCellList{NDIMS}(),
                                       search_radius) where {NDIMS}
    cell_size = ntuple(_ -> search_radius, Val(NDIMS))
    n_cells = ntuple(_ -> -1, Val(NDIMS))

    return FaceNeighborhoodSearch(cell_list, search_radius, nothing, n_cells, cell_size)
end

@inline Base.ndims(::FaceNeighborhoodSearch{NDIMS}) where {NDIMS} = NDIMS

@inline function eachneighbor(coords, neighborhood_search::FaceNeighborhoodSearch)

    return points_in_cell(cell_coords(coords, neighborhood_search), neighborhood_search)
end

faces_in_cell(cell, neighborhood_search) = points_in_cell(cell, neighborhood_search)

function initialize!(neighborhood_search::FaceNeighborhoodSearch, geometry;
                     pad = ntuple(_ -> 1, ndims(geometry)))
    (; cell_list, neighbor_iterator, search_radius) = neighborhood_search

    empty!(cell_list)

    # Fill cells with intersecting faces
    for face in eachface(geometry)

        # Check if any face intersects a cell in the face-embedding cell grid
        for cell in cell_grid(face, geometry, neighborhood_search)
            if cell_intersection(face, geometry, cell, neighborhood_search)
                push_cell!(cell_list, cell, face)
            end
        end
    end

    neighbor_iterator = copy_cell_list(cell_list, search_radius, nothing)
    empty!(neighbor_iterator)

    min_cell = cell_coords(geometry.min_corner, neighborhood_search) .- pad
    max_cell = cell_coords(geometry.max_corner, neighborhood_search) .+ pad

    # Merge all lists of faces in the neighboring cells into one iterator
    face_ids = Int[]
    for cell_runner in meshgrid(min_cell, max_cell)
        resize!(face_ids, 0)
        for neighbor in neighboring_cells(cell_runner, neighborhood_search)
            append!(face_ids, faces_in_cell(Tuple(neighbor), neighborhood_search))
        end

        unique!(face_ids)

        if isempty(face_ids)
            continue
        end

        neighbor_iterator[cell_runner] = copy(face_ids)
    end

    # Copy cell list
    empty!(cell_list)
    for cell in each_cell_index_threadable(neighbor_iterator)
        cell_list[cell] = neighbor_iterator[cell]
    end

    return neighborhood_search
end

function cell_intersection(face, geometry, cell,
                           neighborhood_search::FaceNeighborhoodSearch{NDIMS}) where {NDIMS}
    (; cell_size) = neighborhood_search

    vertices_list = face_vertices(face, geometry)

    # Check if one of the vertices is inside cell
    for v in vertices_list
        cell == cell_coords(v, neighborhood_search) && return true
    end

    # Check if line segments intersect cell
    min_corner = SVector(cell .* cell_size...)
    max_corner = min_corner + SVector(cell_size...)

    ray_direction = vertices_list[2] - vertices_list[1]
    ray_origin = vertices_list[1]

    ray_intersection(min_corner, max_corner, ray_origin, ray_direction) && return true

    if NDIMS == 3
        ray_direction = vertices_list[2] - vertices_list[3]
        ray_origin = vertices_list[3]

        ray_intersection(min_corner, max_corner, ray_origin, ray_direction) && return true

        ray_direction = vertices_list[3] - vertices_list[1]
        ray_origin = vertices_list[1]

        ray_intersection(min_corner, max_corner, ray_origin, ray_direction) && return true

        # For 3D,  Check if triangle plane intersects cell (for very large triangles)
        normal = face_normal(face, geometry)

        return triangle_plane_intersection(ray_origin, normal, min_corner, cell_size)
    end

    return false
end

# See https://tavianator.com/2022/ray_box_boundary.html
function ray_intersection(min_corner, max_corner, ray_origin, ray_direction;
                          pad = sqrt(eps()))
    NDIMS = length(ray_origin)

    inv_dir = SVector(ntuple(@inline(dim->1 / ray_direction[dim]), NDIMS))

    tmin = zero(eltype(ray_direction))
    tmax = Inf
    @inbounds for dim in 1:NDIMS
        # `pad` is to handle rays on the boundary
        t1 = (min_corner[dim] - pad - ray_origin[dim]) * inv_dir[dim]
        t2 = (max_corner[dim] + pad - ray_origin[dim]) * inv_dir[dim]

        tmin = min(max(t1, tmin), max(t2, tmin))
        tmax = max(min(t1, tmax), min(t2, tmax))
    end

    return tmin <= tmax
end

# Check if each cell vertex is located on the same side of the plane.
# Otherwise the plane intersects the cell.
function triangle_plane_intersection(point_on_plane, plane_normal, cell_min_corner,
                                     cell_size)
    cell_center = cell_min_corner .+ cell_size ./ 2

    # `corner1` is the corner that is furthest in the direction of `plane_normal`.
    # When the plane is not intersecting the cell, this corner is the one closest
    # to the plane when the normal is pointing away from the cell or furthest from the plane
    # when the normal is pointing towards the cell.
    # Note that this could also be a face or edge midpoint when the plane is axis-aligned.
    normal_unit = sign.(plane_normal)
    corner1 = cell_center + normal_unit .* cell_size / 2
    corner2 = cell_center - normal_unit .* cell_size / 2

    # These two vectors are on the same side of the plane
    # if and only if the plane intersects the cell
    plane_to_corner1 = corner1 - point_on_plane
    plane_to_corner2 = corner2 - point_on_plane

    # Return true if the two vectors are on different sides of the plane
    return dot(plane_normal, plane_to_corner1) * dot(plane_normal, plane_to_corner2) < 0
end

@inline function cell_grid(face, geometry,
                           neighborhood_search::FaceNeighborhoodSearch{NDIMS}) where {NDIMS}
    vertice_list = face_vertices(face, geometry)

    # Compute the cell coordinates for each vertex
    cells = [cell_coords(v, neighborhood_search) for v in vertice_list]

    # Compute the element-wise minimum and maximum cell coordinates across all vertices
    mins = reduce((a, b) -> min.(a, b), cells)
    maxs = reduce((a, b) -> max.(a, b), cells)

    return meshgrid(mins, maxs)
end

@inline function meshgrid(min_corner, max_corner; increment = 1)
    min_ = collect(min_corner)
    max_ = collect(max_corner)

    ranges = ntuple(dim -> (min_[dim]:increment:max_[dim]), length(min_corner))

    return Iterators.product(ranges...)
end
