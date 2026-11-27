"""
Benchmark `CellListMapNeighborhoodSearch` against `GridNeighborhoodSearch`
(`DictionaryCellList` and `FullGridCellList`) at PointNeighbors.jl's own "typical" SPH
density (`search_radius_factor = 3.0`, the default used throughout [`run_benchmark`](@ref)),
i.e. roughly 27 particles per `search_radius`-sized cell, much sparser than the atomic water
density used in `cell_list_map_water.jl` (~173 particles per cell). This is the density
regime in which the original CellListMap.jl neighborhood search PR
(trixi-framework/PointNeighbors.jl#8) found `FullGridCellList` to outperform CellListMap.jl.

See `cell_list_map_common.jl` for why `SerialBackend()` is used throughout (not just for
comparability, but because `CellListMapNeighborhoodSearch` is not safe with a parallel
backend for the unsynchronized per-point accumulation these benchmarks use).

Run with
```julia
include("benchmarks/cell_list_map_sph.jl")
```
"""

include("cell_list_map_common.jl")

run_sph_benchmark = run_cell_list_map_benchmark(3.0f0)

run_cell_list_map_all_phases(run_sph_benchmark, "SPH-typical density (search_radius_factor=3.0)")
