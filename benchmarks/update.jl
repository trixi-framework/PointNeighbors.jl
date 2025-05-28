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
"""
function benchmark_update_alternating(neighborhood_search, coordinates;
                                      parallelization_backend = default_backend(coordinates))
    coordinates2 = copy(coordinates)
    # Perturb all coordinates with a perturbation factor of `0.015`.
    # This factor was tuned so that ~0.5% of the particles change their cell during an
    # update in 2D and ~0.7% in 3D.
    # These values are the same as the experimentally computed averages in 2D and 3D SPH
    # dam break simulations. So this benchmark replicates a real-life SPH update.
    perturb!(coordinates2, 4e-4 * PointNeighbors.search_radius(neighborhood_search))

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
