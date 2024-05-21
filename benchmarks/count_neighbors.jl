using PointNeighbors
using BenchmarkTools

"""
    benchmark_count_neighbors(neighborhood_search, coordinates; parallel = true)

A very cheap and simple neighborhood search benchmark, only counting the neighbors of each
particle. For each particle-neighbor pair, only an array entry is incremented.

Due to the minimal computational cost, differences between neighborhood search
implementations are highlighted. On the other hand, this is the least realistic benchmark.

For a computationally heavier benchmark, see [`benchmark_n_body`](@ref).
"""
function benchmark_count_neighbors(neighborhood_search, coordinates; parallel = true)
    n_neighbors = zeros(Int, size(coordinates, 2))

    function count_neighbors!(n_neighbors, coordinates, neighborhood_search, parallel)
        n_neighbors .= 0

        for_particle_neighbor(coordinates, coordinates, neighborhood_search,
                              parallel = parallel) do i, _, _, _
            n_neighbors[i] += 1
        end
    end

    return @belapsed $count_neighbors!($n_neighbors, $coordinates,
                                       $neighborhood_search, $parallel)
end
