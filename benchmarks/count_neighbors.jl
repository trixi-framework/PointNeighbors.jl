using PointNeighbors
using BenchmarkTools

"""
    benchmark_count_neighbors(neighborhood_search, coordinates;
                              parallelization_backend = default_backend(coordinates))

A very cheap and simple neighborhood search benchmark, only counting the neighbors of each
point. For each point-neighbor pair, only an array entry is incremented.

Due to the minimal computational cost, differences between neighborhood search
implementations are highlighted. On the other hand, this is the least realistic benchmark.

For a computationally heavier benchmark, see [`benchmark_n_body`](@ref).
"""
function benchmark_count_neighbors(neighborhood_search, coordinates;
                                   parallelization_backend = default_backend(coordinates))
    n_neighbors = zeros(Int, size(coordinates, 2))

    function count_neighbors!(n_neighbors, coordinates, neighborhood_search,
                              parallelization_backend)
        n_neighbors .= 0

        foreach_point_neighbor(coordinates, coordinates, neighborhood_search;
                               parallelization_backend) do i, _, _, _
            n_neighbors[i] += 1
        end
    end

    return @belapsed $count_neighbors!($n_neighbors, $coordinates,
                                       $neighborhood_search, $parallelization_backend)
end
