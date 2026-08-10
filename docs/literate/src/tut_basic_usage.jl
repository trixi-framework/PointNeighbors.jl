# # [Basic Usage: Fixed-Radius Neighbor Search](@id tut_basic_usage)

# This tutorial shows the standard workflow for PointNeighbors.jl:
# create coordinates, configure a neighborhood search, initialize it,
# and loop over neighbors.
using PointNeighbors

# ## Generate a regular 2D point cloud

# We create a regular grid of points in 2D for this example, with distances of 1
# between neighboring points.
# The coordinates need to be stored in a 2×N array, where N is the number of points.
n_points_per_dimension = (100, 100)
n_points = prod(n_points_per_dimension)
coordinates = Array{Float64}(undef, 2, n_points)
cartesian_indices = CartesianIndices(n_points_per_dimension)

for i in axes(coordinates, 2)
    coordinates[:, i] .= Tuple(cartesian_indices[i])
end

# ## Create and initialize the neighborhood search

# We choose a search radius and create the neighborhood search.
# For the [`GridNeighborhoodSearch`](@ref), we need to pass `n_points` as a maximum
# number of points in the neighbor coordinates array, which is required to allocate
# the data structures for the update step.
search_radius = 3.0
nhs = GridNeighborhoodSearch{2}(; search_radius, n_points)
nothing # hide

# Initialize the neighborhood search with the coordinates.
# !!! warning
#     This neighborhood search is now configured for the given coordinates.
#     In general, it is only possible to use this neighborhood search to find neighbors
#     for points contained in `coordinates` and not for arbitrary points in space.
#     See below for more information.
initialize!(nhs, coordinates, coordinates)
nothing # hide

# ## Count neighbors for each point

# With the neighborhood search initialized, we can now loop over neighbors.
# We use the function [`foreach_point_neighbor`](@ref) to loop over all pairs of points
# and neighbors within the search radius.
# !!! warning
#     The `foreach_point_neighbor` function is multithreaded over the points by default.
#     Only remove `parallelization_backend = SerialBackend()` if you are sure that your code
#     is thread-safe and that there are no race conditions.
n_neighbors = zeros(Int, n_points)

function count_neighbors!(n_neighbors, coordinates, neighbor_coordinates, nhs)
    n_neighbors .= 0

    foreach_point_neighbor(coordinates, neighbor_coordinates, nhs,
                           parallelization_backend = SerialBackend()) do i, j,
                                                                         pos_diff, distance
        n_neighbors[i] += 1
    end

    return n_neighbors
end

count_neighbors!(n_neighbors, coordinates, coordinates, nhs)

# Interior particles have many neighbors, boundary particles fewer.
extrema(n_neighbors)

# ## Different point sets for points and neighbors

# The neighborhood search is currently configured to find neighbors in `coordinates`
# for points in `coordinates`. However, it is also possible to find neighbors in a
# different set of coordinates, e.g., `neighbor_coordinates`.
neighbor_coordinates = coordinates .+ 0.5
nothing # hide

# In order to find neighbors in `neighbor_coordinates`, we need to re-initialize
# the neighborhood search with `neighbor_coordinates` as the second argument.
initialize!(nhs, coordinates, neighbor_coordinates)
nothing # hide

# Now the neighborhood search is configured to find neighbors in `neighbor_coordinates`
# for points in `coordinates`.
count_neighbors!(n_neighbors, coordinates, neighbor_coordinates, nhs)
extrema(n_neighbors)

# ## Updating the neighborhood search

# If the coordinates of either the points or the neighbors change, the neighborhood search
# needs to be updated with [`update!`](@ref).
# Depending on the neighborhood search implementation, this can be done more efficiently
# than re-initializing the neighborhood search.
# !!! warning
#     An `update!` requires that the sizes of the point sets do not change.
#
# If we don't update the neighborhood search, we will get incorrect neighbors:
neighbor_coordinates .+= 10
count_neighbors!(n_neighbors, coordinates, neighbor_coordinates, nhs)
extrema(n_neighbors)

# After updating the neighborhood search, we get the correct new neighbors.
update!(nhs, coordinates, neighbor_coordinates)
count_neighbors!(n_neighbors, coordinates, neighbor_coordinates, nhs)
extrema(n_neighbors)

# If the first coordinates are updated but the second coordinates are not, we generally
# also need to update the neighborhood search.
# For some neighborhood search implementations, notably the [`GridNeighborhoodSearch`](@ref),
# this is not necessary, since this implementation can find neighbors for arbitrary points in space.
# To check whether an update is necessary, we can call [`requires_update`](@ref):
requires_update(nhs)

# The first `false` indicates that an update is not required when the first coordinates are
# updated, and the second `true` indicates that an update is required when the second
# coordinates are updated.
