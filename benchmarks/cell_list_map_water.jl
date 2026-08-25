using PointNeighbors
using CellListMap: CellListMap

include("benchmarks.jl")
include("plot_benchmarks.jl")

"""
Benchmark `CellListMapNeighborhoodSearch` against `GridNeighborhoodSearch`
(`DictionaryCellList` and `FullGridCellList`) at a density/cutoff combination typical of
atomistic molecular dynamics of liquid water, rather than PointNeighbors' usual SPH-like
benchmarks:
- Atomic number density of water (counting O and both H per molecule):
  ``\\rho = 0.1003\\, \\mathrm{\\mathring{A}}^{-3}``
  (``\\rho = 1\\,\\mathrm{g/cm^3}``, ``M = 18\\,\\mathrm{g/mol}``, ``3`` atoms/molecule).
- A cutoff of ``12\\,\\mathrm{\\mathring{A}}``, typical of non-bonded MD interactions.

This gives a characteristic interatomic spacing of ``\\rho^{-1/3} \\approx 2.15\\,\\mathrm{\\mathring{A}}``
(close to the real nearest-neighbor spacing in liquid water, a good sanity check) and an
average of ``\\frac{4}{3}\\pi \\cdot 12^3 \\cdot \\rho \\approx 726`` neighbors per particle,
much denser than PointNeighbors' typical SPH benchmarks (~30-60 neighbors per particle).

Run with
```julia
include("benchmarks/cell_list_map_water.jl")
```
"""

# Atomic number density of water in Å⁻³ (O + 2×H per molecule)
const WATER_DENSITY = 0.1003

# Characteristic interatomic spacing at this density
const WATER_SPACING = WATER_DENSITY^(-1 / 3)

# 12 Å cutoff, expressed as a multiple of the interatomic spacing
const CUTOFF = 12.0
const SEARCH_RADIUS_FACTOR = CUTOFF / WATER_SPACING

println("Water benchmark: spacing = $(round(WATER_SPACING, digits = 3)) Å, " *
        "cutoff = $CUTOFF Å, search_radius_factor = $(round(SEARCH_RADIUS_FACTOR, digits = 3)), " *
        "average neighbors/particle ≈ $(round(Int, 4 / 3 * pi * CUTOFF^3 * WATER_DENSITY))")

function run_water_benchmark(benchmark, n_points_per_dimension = (12, 12, 12), iterations = 4;
                             kwargs...)
    NDIMS = length(n_points_per_dimension)
    min_corner = 0.0f0 .* n_points_per_dimension
    max_corner = Float32.(n_points_per_dimension ./ maximum(n_points_per_dimension))

    neighborhood_searches = [
        GridNeighborhoodSearch{NDIMS}(),
        GridNeighborhoodSearch{NDIMS}(search_radius = 0.0f0,
                                      cell_list = FullGridCellList(; search_radius = 0.0f0,
                                                                   min_corner, max_corner)),
        CellListMapNeighborhoodSearch(NDIMS)
    ]

    names = ["GridNeighborhoodSearch (DictionaryCellList)";;
             "GridNeighborhoodSearch (FullGridCellList)";;
             "CellListMapNeighborhoodSearch"]

    run_benchmark(benchmark, n_points_per_dimension, iterations, neighborhood_searches;
                  search_radius_factor = Float32(SEARCH_RADIUS_FACTOR), names, kwargs...)
end

n_particles_count, times_count = run_water_benchmark(benchmark_count_neighbors)
plot_benchmark(n_particles_count, times_count;
              title = "Count neighbors, water density, $(CUTOFF)Å cutoff")

n_particles_nbody, times_nbody = run_water_benchmark(benchmark_n_body)
plot_benchmark(n_particles_nbody, times_nbody;
              title = "N-body, water density, $(CUTOFF)Å cutoff")
