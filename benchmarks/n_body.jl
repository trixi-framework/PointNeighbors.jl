using PointNeighbors
using BenchmarkTools

"""
    benchmark_n_body(neighborhood_search, coordinates;
                     parallelization_backend = default_backend(coordinates))

A simple neighborhood search benchmark, computing the right-hand side of an n-body
simulation with a cutoff (corresponding to the search radius of `neighborhood_search`).

This is a more realistic benchmark for particle-based simulations than
[`benchmark_count_neighbors`](@ref).
However, due to the higher computational cost, differences between neighborhood search
implementations are less pronounced.
"""
function benchmark_n_body(neighborhood_search, coordinates_;
                          parallelization_backend = default_backend(coordinates_))
    # Passing a different backend like `CUDA.CUDABackend`
    # allows us to change the type of the array to run the benchmark on the GPU.
    coordinates = PointNeighbors.Adapt.adapt(parallelization_backend, coordinates_)

    # Remove unnecessary data structures that are only used for initialization
    neighborhood_search_ = PointNeighbors.freeze_neighborhood_search(neighborhood_search)

    nhs = PointNeighbors.Adapt.adapt(parallelization_backend, neighborhood_search_)

    # This preserves the data type of `coordinates`, which makes it work for GPU types
    mass = 1e10 * (rand!(similar(coordinates, size(coordinates, 2))) .+ 1)
    G = 6.6743e-11

    dv = similar(coordinates)

    function compute_acceleration!(dv, coordinates, mass, G, neighborhood_search,
                                   parallelization_backend)
        dv .= 0.0

        foreach_point_neighbor(coordinates, coordinates, neighborhood_search;
                               parallelization_backend) do i, j, pos_diff, distance
            # Only consider particles with a distance > 0
            distance < sqrt(eps()) && return

            dv_ = -G * mass[j] * pos_diff / distance^3

            for dim in axes(dv, 1)
                @inbounds dv[dim, i] += dv_[dim]
            end
        end

        return dv
    end

    return @belapsed $compute_acceleration!($dv, $coordinates, $mass, $G, $nhs,
                                            $parallelization_backend)
end
