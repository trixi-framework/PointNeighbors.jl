# # [N-Body Example with Cutoff Radius](@id tut_n_body)

# This tutorial shows how to compute one right-hand-side evaluation of a simple
# n-body model with a cutoff radius using PointNeighbors.jl.
# It builds on the basic usage tutorial and reuses the same neighbor-loop pattern.
using PointNeighbors
using Random

# ## Generate a simple 2D particle set

# We use uniformly distributed points in 2D. As in all of PointNeighbors.jl,
# coordinates are stored in a 2×N array.
n_particles = 5_000
coordinates = rand(2, n_particles)

# The cutoff radius for pair interactions is the search radius.
search_radius = 0.04
nothing # hide

# Each particle gets a mass in [1e10, 2e10].
mass = 1.0e10 .* (rand(n_particles) .+ 1)
G = Float32(6.6743e-11)
accelerations = similar(coordinates)
nothing # hide

# ## Create and initialize the neighborhood search

nhs = GridNeighborhoodSearch{2}(; search_radius, n_points = n_particles)
initialize!(nhs, coordinates, coordinates)
nothing # hide

# ## Compute one acceleration update

# Sum all pairwise contributions inside the cutoff radius.
# Note that this is a multithreaded loop when starting Julia with multiple threads.
# It is thread-safe because the threading happens over the particles, not the neighbors,
# and each thread only updates the acceleration of its own particle.
function compute_acceleration!(accelerations, coordinates, mass, G, nhs)
    accelerations .= 0.0

    foreach_point_neighbor(coordinates, coordinates, nhs) do i, j, pos_diff, distance
        ## Skip self-interactions. Note that `return` only exits the closure,
        ## i.e., skips the current neighbor.
        distance < eps() && return

        ## `foreach_point_neighbor` makes sure that `i` and `j` are in bounds
        ## of the respective coordinates. This is especially relevant on GPUs,
        ## where bounds checking is more expensive.
        dv = @inbounds -G * mass[j] * pos_diff / distance^3

        for dim in axes(accelerations, 1)
            @inbounds accelerations[dim, i] += dv[dim]
        end
    end

    return accelerations
end

compute_acceleration!(accelerations, coordinates, mass, G, nhs)
nothing # hide

# We now have one acceleration vector per particle.
size(accelerations)
