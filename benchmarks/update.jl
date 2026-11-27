using PointNeighbors
using BenchmarkTools

# For `perturb!`
include("../test/point_cloud.jl")

"""
    benchmark_initialize(neighborhood_search, coordinates;
                         parallelization_backend = default_backend(coordinates))

Benchmark neighborhood search initialization with the given `coordinates`.
"""
function benchmark_initialize(neighborhood_search, coordinates;
                              parallelization_backend = default_backend(coordinates))
    return @belapsed $initialize!($neighborhood_search, $coordinates, $coordinates;
                                  parallelization_backend = $parallelization_backend)
end

"""
    benchmark_update_alternating(neighborhood_search, coordinates;
                                 parallelization_backend = default_backend(coordinates))

A very simple benchmark for neighborhood search update, alternating between two differently
perturbed point clouds.

This is a good benchmark for incremental updates, since most particles stay in their cells.
Similar to SPH dam break simulations, ~0.4% of the particles change their cell during
an update.
"""
function benchmark_update_alternating(neighborhood_search, coordinates;
                                      parallelization_backend = default_backend(coordinates))
    coordinates2 = copy(coordinates)
    # Perturb all coordinates with a standard deviation of
    # `perturbation_factor * search_radius`.
    # These factors were tuned for the (rotated) point clouds of `run_benchmark` with
    # a search radius of 3 times the point spacing, so that ~0.4% of the particles change
    # their cell during an update in 2D and 3D.
    # These are realistic numbers for a dam break simulation (measured with
    # Carpenter-Kennedy and CFL 1.6), averaged over all neighborhood search updates
    # (one per stage of the time integration scheme). The number is independent of the
    # resolution and the same in 2D and 3D.
    # So this benchmark replicates a real-life SPH update.
    perturbation_factor = size(coordinates, 1) == 3 ? 1.7e-3 : 2.5e-3

    # Use the same perturbation every time to make comparisons fair and reproducible.
    rng = Random.Xoshiro(1)
    perturb!(coordinates2,
             perturbation_factor * PointNeighbors.search_radius(neighborhood_search); rng)

    function update_alternating!(neighborhood_search, coordinates, coordinates2,
                                 parallelization_backend)
        update!(neighborhood_search, coordinates, coordinates; parallelization_backend)
        update!(neighborhood_search, coordinates, coordinates2; parallelization_backend)
    end

    result = @belapsed $update_alternating!($neighborhood_search, $coordinates,
                                            $coordinates2, $parallelization_backend)

    # Return average update time
    return 0.5 * result
end
