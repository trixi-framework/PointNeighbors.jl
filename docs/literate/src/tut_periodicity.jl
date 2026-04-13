# # [Periodic Domains with `PeriodicBox`](@id tut_periodicity)

# This tutorial extends the basic setup by adding periodic boundary conditions.
using PointNeighbors

# ## Generate a regular 2D point cloud
n_points_per_dimension = (100, 100)
n_points = prod(n_points_per_dimension)
coordinates = Array{Float64}(undef, 2, n_points)
cartesian_indices = CartesianIndices(n_points_per_dimension)

for i in axes(coordinates, 2)
    coordinates[:, i] .= Tuple(cartesian_indices[i])
end

# ## Configure a periodic domain

# PointNeighbors.jl supports rectangular, axis-aligned periodic domains through the
# [`PeriodicBox`](@ref) type.
search_radius = 3.0
min_corner = (0.0, 0.0)
max_corner = (100.0, 100.0)
periodic_box = PeriodicBox(; min_corner, max_corner)
nothing # hide

# This periodic box can now be passed to the neighborhood search constructor.
nhs = GridNeighborhoodSearch{2}(; search_radius, periodic_box, n_points)
initialize!(nhs, coordinates, coordinates)
nothing # hide

# ## Count neighbors in periodic and non-periodic setups

# This function is the same as in the [basic usage tutorial](@ref tut_basic_usage)
# and counts the number of neighbors for each point.
# Note that this is a multithreaded loop when starting Julia with multiple threads.
# It is thread-safe because the threading happens over the points, not the neighbors,
# and each thread only updates the counter of its own point.
function count_neighbors!(n_neighbors, coordinates, nhs)
    n_neighbors .= 0

    foreach_point_neighbor(coordinates, coordinates, nhs) do i, j, pos_diff, distance
        n_neighbors[i] += 1
    end

    return n_neighbors
end

n_neighbors = zeros(Int, n_points)
count_neighbors!(n_neighbors, coordinates, nhs)
extrema(n_neighbors)

# Since we already have a bounded domain, it makes sense to use a `FullGridCellList`,
# which is focused on maximum performance, but does not support infinite domains.
cell_list = FullGridCellList(; search_radius, min_corner, max_corner)
nhs2 = GridNeighborhoodSearch{2}(; search_radius, periodic_box, cell_list, n_points)
initialize!(nhs2, coordinates, coordinates)
count_neighbors!(n_neighbors, coordinates, nhs2)
extrema(n_neighbors)
