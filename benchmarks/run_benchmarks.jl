using Plots
using BenchmarkTools

# Generate a rectangular point cloud
include("../test/point_cloud.jl")

"""
    plot_benchmarks(benchmark, n_points_per_dimension, iterations;
                    seed = 1, perturbation_factor_position = 1.0,
                    parallel = true, title = "")

Run a benchmark with several neighborhood searches multiple times for increasing numbers
of points and plot the results.

# Arguments
- `benchmark`:              The benchmark function. See [`benchmark_count_neighbors`](@ref)
                            and [`benchmark_n_body`](@ref).
- `n_points_per_dimension`: Initial resolution as tuple. The product is the initial number
                            of points. For example, use `(100, 100)` for a 2D benchmark or
                            `(10, 10, 10)` for a 3D benchmark.
- `iterations`:             Number of refinement iterations

# Keywords
- `parallel = true`:        Loop over all points in parallel
- `title = ""`:             Title of the plot
- `seed = 1`:               Seed to perturb the point positions. Different seeds yield
                            slightly different point positions.
- `perturbation_factor_position = 1.0`: Scale the point position perturbation by this factor.
                                        A factor of `1.0` corresponds to a standard deviation
                                        similar to that of a realistic simulation.

# Examples
```julia
include("benchmarks/benchmarks.jl")

plot_benchmarks(benchmark_count_neighbors, (10, 10), 3)
"""
function run_benchmark(benchmark, n_points_per_dimension, iterations, neighborhood_searches;
                       names = ["NeighborhoodSearch $i"
                                for i in 1:length(neighborhood_searches)]',
                       parallelization_backend = PolyesterBackend(),
                       seed = 1, perturbation_factor_position = 1.0)
    # Multiply number of points in each iteration (roughly) by this factor
    scaling_factor = 4
    per_dimension_factor = scaling_factor^(1 / length(n_points_per_dimension))
    sizes = [round.(Int, n_points_per_dimension .* per_dimension_factor^(iter - 1))
             for iter in 1:iterations]

    n_particles_vec = prod.(sizes)
    times = zeros(iterations, length(neighborhood_searches))

    for iter in 1:iterations
        coordinates = point_cloud(sizes[iter]; seed, perturbation_factor_position)
        domain_size = maximum(sizes[iter]) + 1

        # Normalize domain size to 1
        coordinates ./= domain_size

        # Make this Float32 to make sure that Float32 benchmarks use Float32 exclusively
        search_radius = 4.0f0 / domain_size
        n_particles = size(coordinates, 2)

        neighborhood_searches_copy = copy_neighborhood_search.(neighborhood_searches,
                                                               search_radius, n_particles)

        for i in eachindex(neighborhood_searches_copy)
            neighborhood_search = neighborhood_searches_copy[i]
            PointNeighbors.initialize!(neighborhood_search, coordinates, coordinates)

            time = benchmark(neighborhood_search, coordinates; parallelization_backend)
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

# Rum the benchmark with the most commonly used neighborhood search implementations
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

# Rum the benchmark with all GPU-compatible neighborhood search implementations
function run_benchmark_gpu(benchmark, n_points_per_dimension, iterations; kwargs...)
    NDIMS = length(n_points_per_dimension)

    min_corner = 0.0f0 .* n_points_per_dimension
    max_corner = Float32.(n_points_per_dimension ./ maximum(n_points_per_dimension))
    neighborhood_searches = [
        GridNeighborhoodSearch{NDIMS}(search_radius = 0.0f0,
                                      cell_list = FullGridCellList(; search_radius = 0.0f0,
                                                                   min_corner, max_corner))
    ]

    names = ["GridNeighborhoodSearch with FullGridCellList";;]

    run_benchmark(benchmark, n_points_per_dimension, iterations,
                  neighborhood_searches; names, kwargs...)
end

function plot_benchmark(n_particles_vec, times; kwargs...)
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

    plot(n_particles_vec, times ./ n_particles_vec .* 1e9;
         xaxis = :log,
         xticks = (n_particles_vec, xticks), linewidth = 2,
         xlabel = "#particles", ylabel = "runtime per particle [ns]",
         legend = :outerright, size = (700, 350), dpi = 200, margin = 4 * Plots.mm,
         palette = palette(:tab10), kwargs...)
end
