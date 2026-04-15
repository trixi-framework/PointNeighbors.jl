# # [Advanced Usage: Copying Neighborhood Searches](@id tut_advanced_usage)

using PointNeighbors
# When working with multiple point-sets and therefore multiple neighborhood searches,
# it is often desirable to hide these details from the user-facing API.
# We would like the user to simply pass the type of neighborhood search they want to use,
# and the library should take care of creating the neighborhood searches internally.
#
# For exactly this purpose, PointNeighbors.jl provides the concept of a "template
# neighborhood search", which is a neighborhood search created without essential information
# like search radius or number of points, and the function
# [`copy_neighborhood_search`](@ref), which creates a copy of an existing neighborhood
# search or template, but with a (different) concrete configuration.
#
# For the simplest example, we can work with the [`TrivialNeighborhoodSearch`](@ref),
# which simply loops over all points, resulting in quadratic runtime for finding neighbors
# of a particle.
n_points = 1000
search_radius = 1.0
nhs = TrivialNeighborhoodSearch{2}(; search_radius, eachpoint = 1:n_points)
nothing # hide

# This constructor requires knowledge of the search radius and size of the point set,
# which might differ between different point sets in the same simulation.
# Instead, we can create a template neighborhood search without this information:
template_nhs = TrivialNeighborhoodSearch{2}()
nothing # hide

# This template can now be copied with different configurations for different point sets.
nhs1 = copy_neighborhood_search(template_nhs, search_radius, n_points)
nhs2 = copy_neighborhood_search(template_nhs, search_radius * 2, n_points * 2)
nothing # hide

# The same concept can be applied to all neighborhood search implementations in
# PointNeighbors.jl. All templates can be copied with exactly the same function call.
template_nhs2 = GridNeighborhoodSearch{2}()
nhs3 = copy_neighborhood_search(template_nhs2, search_radius, n_points)
nothing # hide

# Additional configuration options can be passed to the templates and will be preserved
# through the copying process. This applies for example to the periodic box or the
# update strategy of the [`GridNeighborhoodSearch`](@ref).
periodic_box = PeriodicBox(min_corner = (0.0, 0.0), max_corner = (10.0, 10.0))
template_nhs3 = GridNeighborhoodSearch{2}(; periodic_box, update_strategy = SerialUpdate())
nhs4 = copy_neighborhood_search(template_nhs3, search_radius, n_points)
nhs4.update_strategy, nhs4.periodic_box
