using Plots
using BenchmarkTools

# Generate a rectangular point cloud
include("../test/point_cloud.jl")

"""
    run_benchmarks(benchmark, n_points_per_dimension, iterations, neighborhood_searches;
                   search_radius_factor = 3.0,
                   parallelization_backend = PolyesterBackend(),
                   names = ["Neighborhood search 1" "Neighborhood search 2" ...],
                   seed = 1, perturbation_factor_position = 1.0, shuffle = false)

Run a benchmark with several neighborhood searches multiple times for increasing numbers
of points and return the results as `(n_particles_vec, times)`, where `n_particles_vec`
is a vector containing the number of particles for each iteration and `times` is a matrix
containing the runtimes for each neighborhood search and iteration.

See also
- [`plot_benchmark`](@ref) to plot the results,
- [`run_benchmark_default`](@ref) to run the benchmark with the most commonly used
  neighborhood search implementations,
- [`run_benchmark_gpu`](@ref) to run the benchmark with all GPU-compatible neighborhood
  search implementations.

# Arguments
- `benchmark`:              The benchmark function. See [`benchmark_count_neighbors`](@ref),
                            [`benchmark_n_body`](@ref), [`benchmark_wcsph`](@ref),
                            [`benchmark_tlsph`](@ref), and
                            [`benchmark_tlsph_deformation_grad`](@ref).
- `n_points_per_dimension`: Initial resolution as tuple. The product is the initial number
                            of points. For example, use `(100, 100)` for a 2D benchmark or
                            `(10, 10, 10)` for a 3D benchmark.
- `iterations`:             Number of refinement iterations

# Keywords
- `search_radius_factor = 3.0`: Search radius as a multiple of the point spacing.
                            If supported by the benchmark, the type
                            of `search_radius_factor` determines if the benchmark
                            is run in single or double precision.
- `parallelization_backend = PolyesterBackend()`: Parallelization strategy to use. See
                            [`@threaded`](@ref) for a list of available backends.
- `names = ["Neighborhood search 1" ...]`: Names of the neighborhood searches used in the
                            benchmark output.
- `seed = 1`:               Seed to perturb the point positions. Different seeds yield
                            slightly different point positions.
- `perturbation_factor_position = 1.0`: Scale the point position perturbation by this factor.
                            A factor of `1.0` corresponds to a standard deviation
                            similar to that of a realistic simulation.
- `shuffle = false`:        Randomly shuffle the point ordering instead of sorting points by
                            cell index.

# Examples
```julia
include("benchmarks/benchmarks.jl")

run_benchmark(benchmark_count_neighbors, (10, 10), 3,
              [TrivialNeighborhoodSearch{2}(), GridNeighborhoodSearch{2}()])
```
"""
function run_benchmark(benchmark, n_points_per_dimension, iterations, neighborhood_searches;
                       search_radius_factor = 3.0,
                       parallelization_backend = PolyesterBackend(),
                       names = ["Neighborhood search $i"
                                for i in 1:length(neighborhood_searches)]',
                       seed = 1, perturbation_factor_position = 1.0, shuffle = false)
    if !(search_radius_factor isa AbstractFloat && isfinite(search_radius_factor) &&
         search_radius_factor > 0)
        throw(ArgumentError("`search_radius_factor` must be a finite, positive float"))
    end

    # Multiply number of points in each iteration (roughly) by this factor
    scaling_factor = 4
    per_dimension_factor = scaling_factor^(1 / length(n_points_per_dimension))
    sizes = [round.(Int, n_points_per_dimension .* per_dimension_factor^(iter - 1))
             for iter in 1:iterations]

    n_particles_vec = prod.(sizes)
    times = zeros(iterations, length(neighborhood_searches))

    for iter in 1:iterations
        coordinates_ = point_cloud(sizes[iter], search_radius_factor;
                                   seed, perturbation_factor_position, shuffle)
        coordinates = convert.(typeof(search_radius_factor), coordinates_)
        domain_size = maximum(sizes[iter]) + 1

        # Normalize domain size to 1
        coordinates ./= domain_size

        search_radius = search_radius_factor / domain_size
        n_particles = size(coordinates, 2)

        neighborhood_searches_copy = copy_neighborhood_search.(neighborhood_searches,
                                                               search_radius, n_particles)

        for i in eachindex(neighborhood_searches_copy)
            neighborhood_search_ = neighborhood_searches_copy[i]
            neighborhood_search = PointNeighbors.Adapt.adapt(parallelization_backend,
                                                             neighborhood_search_)
            coords = PointNeighbors.Adapt.adapt(parallelization_backend, coordinates)
            PointNeighbors.initialize!(neighborhood_search, coords, coords)

            time = benchmark(neighborhood_search, coords; parallelization_backend)
            times[iter, i] = time
            time_string = BenchmarkTools.prettytime(time * 1e9)
            time_string_per_particle = BenchmarkTools.prettytime(time * 1e9 / n_particles)
            println("$(names[i])")
            println("with $(join(sizes[iter], "x")) = $(prod(sizes[iter])) particles " *
                    "finished in $time_string ($time_string_per_particle per particle)\n")
        end
    end

    return n_particles_vec, times
end

"""
    run_benchmark_default(benchmark, n_points_per_dimension, iterations; kwargs...)

Shortcut to call [`run_benchmark`](@ref) with the most commonly used neighborhood search
implementations:
- `GridNeighborhoodSearch`
- `GridNeighborhoodSearch` with `FullGridCellList`
- `PrecomputedNeighborhoodSearch`

# Arguments
- `benchmark`:              The benchmark function. See [`benchmark_count_neighbors`](@ref),
                            [`benchmark_n_body`](@ref), [`benchmark_wcsph`](@ref),
                            [`benchmark_tlsph`](@ref), and
                            [`benchmark_tlsph_deformation_grad`](@ref).
- `n_points_per_dimension`: Initial resolution as tuple. The product is the initial number
                            of points. For example, use `(100, 100)` for a 2D benchmark or
                            `(10, 10, 10)` for a 3D benchmark.
- `iterations`:             Number of refinement iterations

# Keywords
See [`run_benchmark`](@ref) for a list of available keywords.

# Examples
```julia
include("benchmarks/benchmarks.jl")

run_benchmark_default(benchmark_n_body, (10, 10), 3)
```
"""
function run_benchmark_default(benchmark, n_points_per_dimension, iterations; kwargs...)
    NDIMS = length(n_points_per_dimension)
    min_corner = 0.0f0 .* n_points_per_dimension
    max_corner = Float32.(n_points_per_dimension ./ maximum(n_points_per_dimension))

    neighborhood_searches = [
        GridNeighborhoodSearch{NDIMS}(),
        GridNeighborhoodSearch{NDIMS}(search_radius = 0.0f0,
                                      cell_list = FullGridCellList(; search_radius = 0.0f0,
                                                                   min_corner, max_corner)),
        PrecomputedNeighborhoodSearch{NDIMS}()
    ]

    names = ["GridNeighborhoodSearch";;
             "GridNeighborhoodSearch with FullGridCellList";;
             "PrecomputedNeighborhoodSearch"]

    run_benchmark(benchmark, n_points_per_dimension, iterations,
                  neighborhood_searches; names, kwargs...)
end

"""
    run_benchmark_gpu(benchmark, n_points_per_dimension, iterations; kwargs...)

Shortcut to call [`run_benchmark`](@ref) with all GPU-compatible neighborhood search
implementations:
- `GridNeighborhoodSearch` with `FullGridCellList`
- `PrecomputedNeighborhoodSearch`

# Arguments
- `benchmark`:              The benchmark function. See [`benchmark_count_neighbors`](@ref),
                            [`benchmark_n_body`](@ref), [`benchmark_wcsph`](@ref),
                            [`benchmark_tlsph`](@ref), and
                            [`benchmark_tlsph_deformation_grad`](@ref).
- `n_points_per_dimension`: Initial resolution as tuple. The product is the initial number
                            of points. For example, use `(100, 100)` for a 2D benchmark or
                            `(10, 10, 10)` for a 3D benchmark.
- `iterations`:             Number of refinement iterations

# Keywords
See [`run_benchmark`](@ref) for a list of available keywords.

# Examples
```julia
include("benchmarks/benchmarks.jl")

run_benchmark_gpu(benchmark_n_body, (10, 10), 3)
```
"""
function run_benchmark_gpu(benchmark, n_points_per_dimension, iterations;
                           parallelization_backend = PolyesterBackend(), kwargs...)
    grid_nhs = create_full_grid_neighborhood_search(n_points_per_dimension)
    precomputed_nhs = create_precomputed_neighborhood_search(grid_nhs,
                                                             parallelization_backend)
    neighborhood_searches = (grid_nhs, precomputed_nhs)

    names = ["GridNeighborhoodSearch with FullGridCellList";;
             "PrecomputedNeighborhoodSearch"]

    run_benchmark(benchmark, n_points_per_dimension, iterations,
                  neighborhood_searches; names, parallelization_backend, kwargs...)
end

"""
    run_benchmark_full_grid(benchmark, n_points_per_dimension, iterations; kwargs...)

Shortcut to call [`run_benchmark`](@ref) with a `GridNeighborhoodSearch` with a
`FullGridCellList`. This is the neighborhood search implementation that is used
in TrixiParticles.jl when performance is important.
Use this function to benchmark and profile TrixiParticles.jl kernels.

# Arguments
- `benchmark`:              The benchmark function. See [`benchmark_count_neighbors`](@ref),
                            [`benchmark_n_body`](@ref), [`benchmark_wcsph`](@ref),
                            [`benchmark_tlsph`](@ref), and
                            [`benchmark_tlsph_deformation_grad`](@ref).
- `n_points_per_dimension`: Initial resolution as tuple. The product is the initial number
                            of points. For example, use `(100, 100)` for a 2D benchmark or
                            `(10, 10, 10)` for a 3D benchmark.
- `iterations`:             Number of refinement iterations

# Keywords
See [`run_benchmark`](@ref) for a list of available keywords.

# Examples
```julia
include("benchmarks/benchmarks.jl")

run_benchmark_full_grid(benchmark_n_body, (10, 10), 3)
```
"""
function run_benchmark_full_grid(benchmark, n_points_per_dimension, iterations;
                                 parallelization_backend = PolyesterBackend(), kwargs...)
    neighborhood_searches = (create_full_grid_neighborhood_search(n_points_per_dimension),)

    names = ["GridNeighborhoodSearch with FullGridCellList";;]

    run_benchmark(benchmark, n_points_per_dimension, iterations,
                  neighborhood_searches; names, parallelization_backend, kwargs...)
end

"""
    run_benchmark_precomputed(benchmark, n_points_per_dimension, iterations; kwargs...)

Shortcut to call [`run_benchmark`](@ref) with a `PrecomputedNeighborhoodSearch`.
This is the neighborhood search implementation that is used in TrixiParticles.jl for
Total Lagrangian SPH, where the neighborhood is computed in initial coordinates.
Use this function to benchmark and profile TrixiParticles.jl kernels.

# Arguments
- `benchmark`:              The benchmark function. See [`benchmark_count_neighbors`](@ref),
                            [`benchmark_n_body`](@ref), [`benchmark_wcsph`](@ref),
                            [`benchmark_tlsph`](@ref), and
                            [`benchmark_tlsph_deformation_grad`](@ref).
- `n_points_per_dimension`: Initial resolution as tuple. The product is the initial number
                            of points. For example, use `(100, 100)` for a 2D benchmark or
                            `(10, 10, 10)` for a 3D benchmark.
- `iterations`:             Number of refinement iterations

# Keywords
See [`run_benchmark`](@ref) for a list of available keywords.

# Examples
```julia
include("benchmarks/benchmarks.jl")

run_benchmark_precomputed(benchmark_n_body, (10, 10), 3)
```
"""
function run_benchmark_precomputed(benchmark, n_points_per_dimension, iterations;
                                   parallelization_backend = PolyesterBackend(), kwargs...)
    grid_nhs = create_full_grid_neighborhood_search(n_points_per_dimension)
    precomputed_nhs = create_precomputed_neighborhood_search(grid_nhs,
                                                             parallelization_backend)
    neighborhood_searches = (precomputed_nhs,)

    names = ["PrecomputedNeighborhoodSearch";;]

    run_benchmark(benchmark, n_points_per_dimension, iterations,
                  neighborhood_searches; names, parallelization_backend, kwargs...)
end

function create_full_grid_neighborhood_search(n_points_per_dimension)
    NDIMS = length(n_points_per_dimension)

    min_corner = 0.0f0 .* n_points_per_dimension
    max_corner = Float32.(n_points_per_dimension ./ maximum(n_points_per_dimension))
    cell_list = FullGridCellList(; search_radius = 0.0f0, min_corner, max_corner)
    return GridNeighborhoodSearch{NDIMS}(; search_radius = 0.0f0, cell_list,
                                         update_strategy = ParallelUpdate())
end

function create_precomputed_neighborhood_search(grid_nhs, parallelization_backend)
    NDIMS = ndims(grid_nhs)
    transpose_backend = parallelization_backend isa PointNeighbors.KernelAbstractions.GPU
    return PrecomputedNeighborhoodSearch{NDIMS}(; search_radius = 0.0f0,
                                                max_neighbors = 128,
                                                update_neighborhood_search = grid_nhs,
                                                transpose_backend)
end

"""
    plot_benchmark(n_particles_vec, times; kwargs...)

Plot the results of a benchmark run with [`run_benchmark`](@ref).
Note that the arguments are the outputs of that function.

# Arguments
- `n_particles_vec`: Vector containing the number of particles for each iteration.
- `times`:           Matrix containing the runtimes for each neighborhood search and iteration.

# Keywords
Keyword arguments are passed to `Plots.plot`. For example, use `title = "My title"`.

# Examples
```julia
include("benchmarks/benchmarks.jl")

n_particles_vec, times = run_benchmark_default(benchmark_count_neighbors, (10, 10), 3)
plot_benchmark(n_particles_vec, times; title = "Count neighbors benchmark")
```
"""
function plot_benchmark(n_particles_vec, times; kwargs...)
    p = plot()
    plot_benchmark!(p, n_particles_vec, times; kwargs...)
end

function plot_benchmark!(p, n_particles_vec, times; kwargs...)
    function format_n_particles(n)
        if n >= 1_000_000
            return "$(round(Int, n / 1_000_000))M"
        elseif n >= 1_000
            return "$(round(Int, n / 1_000))k"
        else
            return string(n)
        end
    end
    xticks = format_n_particles.(n_particles_vec)

    plot!(p, n_particles_vec, n_particles_vec ./ times .* 1e-6;
          xaxis = :log, xticks = (n_particles_vec, xticks), linewidth = 2,
          xlabel = "#particles", ylabel = "million particle updates per second",
          legend = :outerright, size = (700, 350), dpi = 600, margin = 4 * Plots.mm,
          palette = palette(:tab10), kwargs...)
end
