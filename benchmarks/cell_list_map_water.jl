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

All three phases of neighborhood search usage are benchmarked separately: building the lists
(`initialize!`), updating them for a realistic small perturbation (`update!`), and mapping
over neighbor pairs (`foreach_point_neighbor`, both a cheap and a heavier callback). See
`cell_list_map_common.jl` for why `SerialBackend()` is used throughout (not just for
comparability, but because `CellListMapNeighborhoodSearch` is not safe with a parallel
backend for the unsynchronized per-point accumulation these benchmarks use).

Run with
```julia
include("benchmarks/cell_list_map_water.jl")
```
"""

include("cell_list_map_common.jl")

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

# Average occupancy of a `search_radius`-sized (cube) cell is `WATER_DENSITY * CUTOFF^3`
# (≈173 here). `FullGridCellList`'s default `max_points_per_cell = 100` silently drops
# points beyond that limit instead of erroring (the bounds check that would catch this is
# skipped in the parallel update for performance), so it must be raised for this density;
# a 4x safety margin comfortably covers the Poisson fluctuations across many cells.
const MAX_POINTS_PER_CELL = round(Int, 4 * WATER_DENSITY * CUTOFF^3)

run_water_benchmark = run_cell_list_map_benchmark(Float32(SEARCH_RADIUS_FACTOR);
                                                  max_points_per_cell = MAX_POINTS_PER_CELL)

run_cell_list_map_all_phases(run_water_benchmark, "water density, $(CUTOFF)Å cutoff")
