# Benchmark Suite

```@meta
CurrentModule = Main
```

PointNeighbors.jl includes a benchmark suite for comparing neighborhood search
implementations and update strategies. From the repository root, load it with

```julia
include(joinpath(pkgdir(PointNeighbors),  "benchmarks", "benchmarks.jl"));
```

## Benchmark Runners

```@docs
run_benchmark
run_benchmark_default
run_benchmark_gpu
run_benchmark_full_grid
run_benchmark_precomputed
run_benchmark_updates
```

## Benchmark Workloads

```@docs
benchmark_count_neighbors
benchmark_n_body
benchmark_wcsph
benchmark_tlsph
benchmark_tlsph_deformation_grad
benchmark_initialize
benchmark_update_alternating
```

## Plotting

Load the plotting utilities with

```julia
include(joinpath(pkgdir(PointNeighbors), "benchmarks", "plot_benchmarks.jl"));
```

```@docs
plot_benchmark
```
