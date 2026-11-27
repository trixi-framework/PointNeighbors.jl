using PointNeighbors
using CellListMap: CellListMap

include("benchmarks.jl")
include("plot_benchmarks.jl")

# `CellListMapNeighborhoodSearch` parallelizes by CellListMap.jl's own cell-pair batches,
# not by splitting the `points` loop like every other `AbstractNeighborhoodSearch`. This means
# a given point index can be processed by more than one thread concurrently, which is unsafe
# for the unsynchronized per-point accumulation used by every benchmark here (and by
# essentially all real PointNeighbors.jl/TrixiParticles.jl code, e.g. `dv[:, i] += ...`).
# See the "Parallelization is not point-partitioned" warning on `CellListMapNeighborhoodSearch`'s
# docstring. All benchmarks below therefore explicitly force `SerialBackend()`; this is NOT
# just a performance choice, it is required for correctness with this implementation.

"""
    run_cell_list_map_benchmark(search_radius_factor; max_points_per_cell = 100)

Return a function comparing `GridNeighborhoodSearch` (`DictionaryCellList` and
`FullGridCellList`) against `CellListMapNeighborhoodSearch` at the given
`search_radius_factor`, using [`run_benchmark`](@ref) with `SerialBackend()` (required for
correctness, see above).
"""
function run_cell_list_map_benchmark(search_radius_factor; max_points_per_cell = 100)
    function run(benchmark, n_points_per_dimension = (12, 12, 12), iterations = 4; kwargs...)
        NDIMS = length(n_points_per_dimension)
        min_corner = 0.0f0 .* n_points_per_dimension
        max_corner = Float32.(n_points_per_dimension ./ maximum(n_points_per_dimension))

        neighborhood_searches = [
            GridNeighborhoodSearch{NDIMS}(),
            GridNeighborhoodSearch{NDIMS}(search_radius = 0.0f0,
                                          cell_list = FullGridCellList(; search_radius = 0.0f0,
                                                                       min_corner, max_corner,
                                                                       max_points_per_cell)),
            CellListMapNeighborhoodSearch(NDIMS)
        ]

        names = ["GridNeighborhoodSearch (DictionaryCellList)";;
                 "GridNeighborhoodSearch (FullGridCellList)";;
                 "CellListMapNeighborhoodSearch"]

        run_benchmark(benchmark, n_points_per_dimension, iterations, neighborhood_searches;
                      search_radius_factor, names,
                      parallelization_backend = SerialBackend(), kwargs...)
    end
end

"""
    run_cell_list_map_all_phases(run, title_suffix)

Run and plot all three benchmarked phases (build, update, mapping) with `run`
(as returned by [`run_cell_list_map_benchmark`](@ref)).
"""
function run_cell_list_map_all_phases(run, title_suffix)
    println("=== 1. Building the lists (benchmark_initialize) ===")
    n, t = run(benchmark_initialize)
    plot_benchmark(n, t; title = "Initialize, $title_suffix")

    println("=== 2. Updating (benchmark_update_alternating) ===")
    n, t = run(benchmark_update_alternating)
    plot_benchmark(n, t; title = "Update, $title_suffix")

    println("=== 3. Mapping (benchmark_count_neighbors) ===")
    n, t = run(benchmark_count_neighbors)
    plot_benchmark(n, t; title = "Count neighbors, $title_suffix")

    println("=== 3. Mapping (benchmark_n_body) ===")
    n, t = run(benchmark_n_body)
    plot_benchmark(n, t; title = "N-body, $title_suffix")
end
