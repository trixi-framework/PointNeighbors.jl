using PointNeighbors
using BenchmarkTools

"""
    benchmark_n_body(neighborhood_search, coordinates; parallel = true)

A simple neighborhood search benchmark, computing the right-hand side of an n-body
simulation with a cutoff (corresponding to the search radius of `neighborhood_search`).

This is a more realistic benchmark for particle-based simulations than
[`benchmark_count_neighbors`](@ref).
However, due to the higher computational cost, differences between neighborhood search
implementations are less pronounced.
"""
function benchmark_n_body(neighborhood_search, coordinates; parallel = true)
    mass = 1e10 * (rand(size(coordinates, 2)) .+ 1)
    G = 6.6743e-11

    dv = similar(coordinates)

    function compute_acceleration!(dv, coordinates, mass, G, neighborhood_search, parallel)
        dv .= 0.0

        for_particle_neighbor(coordinates, coordinates, neighborhood_search,
                              parallel = parallel) do i, j, pos_diff, distance
            # Only consider particles with a distance > 0
            distance < sqrt(eps()) && return

            dv_ = -G * mass[j] * pos_diff / distance^3

            for dim in axes(dv, 1)
                @inbounds dv[dim, i] += dv_[dim]
            end
        end

        return dv
    end

    return @belapsed $compute_acceleration!($dv, $coordinates, $mass, $G,
                                            $neighborhood_search, $parallel)
end
