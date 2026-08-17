# Benchmark Suite

```@meta
CurrentModule = Main
```

PointNeighbors.jl includes a benchmark suite for comparing neighborhood search
implementations and update strategies. From the repository root, load it with

```julia
using PointNeighbors
include(joinpath(pkgdir(PointNeighbors),  "benchmarks", "benchmarks.jl"));
```

## Benchmark Runners

```@autodocs
Modules = [Main]
Pages = ["run_benchmarks.jl"]
Order = [:function]
```

## Benchmark Workloads

```@autodocs
Modules = [Main]
Pages = ["count_neighbors.jl", "n_body.jl",
         "smoothed_particle_hydrodynamics.jl", "update.jl"]
Order = [:function]
```

## Plotting

Load the plotting utilities with

```julia
using PointNeighbors
include(joinpath(pkgdir(PointNeighbors), "benchmarks", "plot_benchmarks.jl"));
```

```@autodocs
Modules = [Main]
Pages = ["plot_benchmarks.jl"]
Order = [:function]
```
