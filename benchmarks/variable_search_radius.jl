using PointNeighbors
using BenchmarkTools
using Random

# Benchmarks with a variable search radius `h_i` for each point `i`, where two points
# `i` and `j` are neighbors if their distance is at most the pair search radius defined by
# the neighbor criterion (see `TreeNeighborhoodSearch`), for example `max(h_i, h_j)`.
#
# These benchmarks compare the `TreeNeighborhoodSearch` with a `GridNeighborhoodSearch`
# using the maximum search radius, where the neighbors are filtered
# by the pair search radius inside the neighbor loop.

# Loop over all neighbors of all points with the pair search radius.
# With the `TreeNeighborhoodSearch`, each point is queried with its own search radius,
# and the neighbor criterion of the neighborhood search is used.
function foreach_point_neighbor_variable_radius(f::T, coordinates, search_radii,
                                                neighborhood_search::TreeNeighborhoodSearch;
                                                parallelization_backend = default_backend(coordinates),
                                                neighbor_criterion = nothing) where {T}
    # The type annotation above is to make Julia specialize on the type of the function,
    # which is only passed through to `foreach_neighbor`. See `foreach_point_neighbor`.
    PointNeighbors.@threaded parallelization_backend for point in axes(coordinates, 2)
        @inbounds foreach_neighbor(f, coordinates, coordinates, neighborhood_search, point,
                                   search_radius = search_radii[point])
    end

    return nothing
end

# With other neighborhood searches, the search radius of the neighborhood search
# is the maximum search radius, and all neighbors are filtered by the pair search radius.
function foreach_point_neighbor_variable_radius(f::T, coordinates, search_radii,
                                                neighborhood_search;
                                                parallelization_backend = default_backend(coordinates),
                                                neighbor_criterion = MaximumRadius()) where {T}
    foreach_point_neighbor(coordinates, coordinates, neighborhood_search;
                           parallelization_backend) do i, j, pos_diff, distance
        if distance <= PointNeighbors.pair_radius(neighbor_criterion, search_radii[i],
                                      search_radii[j])
            @inline f(i, j, pos_diff, distance)
        end
    end

    return nothing
end

function initialize_variable_radius!(neighborhood_search::TreeNeighborhoodSearch,
                                     coordinates, search_radii;
                                     parallelization_backend = default_backend(coordinates))
    initialize_tree!(neighborhood_search, coordinates, search_radii;
                     parallelization_backend)
end

function initialize_variable_radius!(neighborhood_search, coordinates, search_radii;
                                     parallelization_backend = default_backend(coordinates))
    initialize!(neighborhood_search, coordinates, coordinates; parallelization_backend)
end

"""
    benchmark_count_neighbors_variable_radius(neighborhood_search, coordinates,
                                              search_radii;
                                              parallelization_backend = default_backend(coordinates),
                                              neighbor_criterion = MaximumRadius())

Like [`benchmark_count_neighbors`](@ref), but with a variable search radius for each point.
The `neighbor_criterion` is used to filter the neighbors of neighborhood searches
other than the [`TreeNeighborhoodSearch`](@ref), which uses its own neighbor criterion.
"""
function benchmark_count_neighbors_variable_radius(neighborhood_search, coordinates,
                                                   search_radii;
                                                   parallelization_backend = default_backend(coordinates),
                                                   neighbor_criterion = MaximumRadius())
    # Allocate the neighbor counts on the same device as the coordinates
    n_neighbors = similar(coordinates, Int, size(coordinates, 2))

    function count_neighbors!(n_neighbors, coordinates, search_radii, neighborhood_search,
                              parallelization_backend, neighbor_criterion)
        n_neighbors .= 0

        foreach_point_neighbor_variable_radius(coordinates, search_radii,
                                               neighborhood_search;
                                               parallelization_backend,
                                               neighbor_criterion) do i, _, _, _
            @inbounds n_neighbors[i] += 1
        end
    end

    return @belapsed $count_neighbors!($n_neighbors, $coordinates, $search_radii,
                                       $neighborhood_search, $parallelization_backend,
                                       $neighbor_criterion)
end

"""
    benchmark_initialize_variable_radius(neighborhood_search, coordinates, search_radii;
                                         parallelization_backend = default_backend(coordinates))

Benchmark the initialization of a neighborhood search with a variable search radius
for each point.
"""
function benchmark_initialize_variable_radius(neighborhood_search, coordinates,
                                              search_radii;
                                              parallelization_backend = default_backend(coordinates),
                                              neighbor_criterion = nothing)
    return @belapsed $initialize_variable_radius!($neighborhood_search, $coordinates,
                                                  $search_radii;
                                                  parallelization_backend = $parallelization_backend)
end

"""
    refined_point_cloud(n_points_per_dimension, radius_variation, search_radius_factor;
                        fine_region_fraction = 1 / 3, seed = 1,
                        perturbation_factor_position = 1.0, shuffle = false)

Generate a point cloud with a variable point spacing, similar to a simulation with adaptive
particle refinement, and return the coordinates and the search radii.

The domain is divided in x-direction into
- a fine region with the point spacing `dp_fine`, covering the first `fine_region_fraction`
  of the domain,
- a transition region of length `radius_variation * h_max`, in which the point spacing
  grows linearly from `dp_fine` to `dp_coarse = radius_variation * dp_fine`,
- a coarse region with the point spacing `dp_coarse`.

The points are placed in planes (columns in 2D) perpendicular to the x-axis.
In each plane, the points form a regular grid with the local point spacing, which is
also the distance to the next plane.
The search radius of each point is `search_radius_factor` times the local point spacing,
so `h_max = search_radius_factor * dp_coarse`.

`n_points_per_dimension` is the number of points per dimension of a coarse grid
covering the domain, so the coarse point spacing is `1 / maximum(n_points_per_dimension)`.
"""
function refined_point_cloud(n_points_per_dimension, radius_variation,
                             search_radius_factor; fine_region_fraction = 1 / 3,
                             seed = 1, perturbation_factor_position = 1.0,
                             shuffle = false)
    # Fixed seed to ensure reproducibility
    Random.seed!(seed)

    NDIMS = length(n_points_per_dimension)
    coarse_spacing = 1 / maximum(n_points_per_dimension)
    fine_spacing = coarse_spacing / radius_variation
    domain_size = n_points_per_dimension .* coarse_spacing

    fine_region_end = fine_region_fraction * domain_size[1]
    # Use a longer transition region for larger ratios to keep the gradient of the point
    # spacing small
    transition_length = radius_variation * search_radius_factor * coarse_spacing

    coordinates = Vector{SVector{NDIMS, Float64}}()
    search_radii = Float64[]

    # Add planes of points perpendicular to the x-axis until the domain is filled
    plane_start = 0.0
    while plane_start < domain_size[1] - coarse_spacing / 2
        # The local point spacing at the start of the plane
        if plane_start < fine_region_end
            spacing = fine_spacing
        else
            ramp = min((plane_start - fine_region_end) / transition_length, 1)
            spacing = fine_spacing + ramp * (coarse_spacing - fine_spacing)
        end

        # Regular grid with the local point spacing in the other dimensions
        n_points_plane = ntuple(dim -> max(round(Int, domain_size[dim + 1] / spacing), 1),
                                NDIMS - 1)
        for cartesian_index in CartesianIndices(n_points_plane)
            other_coords = ntuple(dim -> (cartesian_index[dim] - 0.5) *
                                         domain_size[dim + 1] / n_points_plane[dim],
                                  NDIMS - 1)
            point_coords = SVector(plane_start + spacing / 2, other_coords...)

            # Same perturbation relative to the point spacing as in `point_cloud`
            point_coords += perturbation_factor_position * 0.05 * spacing *
                            randn(SVector{NDIMS, Float64})

            push!(coordinates, point_coords)
            push!(search_radii, search_radius_factor * spacing)
        end

        plane_start += spacing
    end

    # Sort by the cells of the largest search radius like in `point_cloud`, or randomly
    if shuffle
        permutation = Random.shuffle(eachindex(coordinates))
    else
        cell_size = ntuple(_ -> search_radius_factor * coarse_spacing, NDIMS)
        cell_coords = [PointNeighbors.nonperiodic_cell_coords(x, nothing, cell_size)
                       for x in coordinates]
        permutation = sortperm(cell_coords)
    end

    return stack(coordinates[permutation]), search_radii[permutation]
end

"""
    run_benchmark_variable_radius(benchmark, n_points_per_dimension, iterations;
                                  radius_variation = 4.0, search_radius_factor = 3.0,
                                  fine_region_fraction = 1 / 3,
                                  neighbor_criterion = MaximumRadius(),
                                  parallelization_backend = PolyesterBackend(),
                                  seed = 1, perturbation_factor_position = 1.0,
                                  shuffle = false)

Run a benchmark with a variable search radius for each point multiple times for increasing
numbers of points and return the results as `(n_particles_vec, times)`.
See [`run_benchmark`](@ref) for more details.

The points are generated with [`refined_point_cloud`](@ref), which has a fine region,
a transition region and a coarse region with a point spacing of `radius_variation` times
the fine point spacing.
The search radius of each point is `search_radius_factor` times the local point spacing.
Two points `i` and `j` are neighbors if their distance is at most the pair search radius
defined by `neighbor_criterion` (see [`TreeNeighborhoodSearch`](@ref)).

The following neighborhood searches are compared:
- `TreeNeighborhoodSearch` without and with pruning, querying each point with its own
  search radius,
- `TreeNeighborhoodSearch` with pruning and `CompactVectorOfVectors`, which also skips
  parts of large leaves,
- `GridNeighborhoodSearch` with `FullGridCellList` with the maximum search radius,
  filtering the neighbors by the pair search radius inside the neighbor loop.

# Arguments
- `benchmark`:              The benchmark function.
                            See [`benchmark_count_neighbors_variable_radius`](@ref) and
                            [`benchmark_initialize_variable_radius`](@ref).
- `n_points_per_dimension`: Initial resolution of the coarse region as tuple.
                            See [`refined_point_cloud`](@ref).
                            For example, use `(100, 100)` for a 2D benchmark or
                            `(10, 10, 10)` for a 3D benchmark.
- `iterations`:             Number of refinement iterations

# Keywords
- `radius_variation = 4.0`:         Ratio of the coarse and the fine point spacing,
                                    which is also the ratio of the maximum and the minimum
                                    search radius. Use `1.0` for a uniform search radius.
- `fine_region_fraction = 1 / 3`:   Fraction of the domain length in x-direction
                                    covered by the fine region.
- `neighbor_criterion = MaximumRadius()`: Criterion defining which points are neighbors.
                                    See [`TreeNeighborhoodSearch`](@ref).
See [`run_benchmark`](@ref) for the other keywords.

# Examples
```julia
include("benchmarks/benchmarks.jl")

run_benchmark_variable_radius(benchmark_count_neighbors_variable_radius, (100, 100), 3)
```
"""
function run_benchmark_variable_radius(benchmark, n_points_per_dimension, iterations;
                                       radius_variation = 4.0, search_radius_factor = 3.0,
                                       fine_region_fraction = 1 / 3,
                                       neighbor_criterion = MaximumRadius(),
                                       parallelization_backend = PolyesterBackend(),
                                       seed = 1, perturbation_factor_position = 1.0,
                                       shuffle = false)
    NDIMS = length(n_points_per_dimension)
    ELTYPE = typeof(search_radius_factor)

    # Multiply number of points in each iteration (roughly) by this factor
    scaling_factor = 4
    per_dimension_factor = scaling_factor^(1 / NDIMS)
    sizes = [round.(Int, n_points_per_dimension .* per_dimension_factor^(iter - 1))
             for iter in 1:iterations]

    n_particles_vec = zeros(Int, iterations)
    names = ["TreeNeighborhoodSearch";;
             "TreeNeighborhoodSearch with pruning";;
             "TreeNeighborhoodSearch with pruning and CompactVectorOfVectors";;
             "GridNeighborhoodSearch with FullGridCellList and maximum radius"]
    times = zeros(iterations, length(names))

    for iter in 1:iterations
        coordinates_,
        search_radii_ = refined_point_cloud(sizes[iter], radius_variation,
                                            search_radius_factor;
                                            fine_region_fraction, seed,
                                            perturbation_factor_position,
                                            shuffle)
        coordinates = convert.(ELTYPE, coordinates_)
        search_radii = convert.(ELTYPE, search_radii_)
        n_points = n_particles_vec[iter] = size(coordinates, 2)

        min_corner = Tuple(minimum(coordinates, dims = 2))
        max_corner = Tuple(maximum(coordinates, dims = 2))
        min_radius, max_radius = extrema(search_radii)

        # Use root cells of the size of the maximum radius. Then, level 0 of the tree
        # is identical to the grid of the `GridNeighborhoodSearch` below.
        # The finest useful level of the tree has cells of the size of the minimum radius,
        # since cells are only split if their children are larger than all radii.
        root_cell_size = max_radius
        max_level = max(floor(Int, log2(root_cell_size / min_radius)), 0)

        # The cells of the grid have the size of the maximum radius,
        # so they can contain many fine points.
        # Since a cell of the tree is only split if all radii in the surrounding cells
        # are small enough, the fine region next to the transition region is covered by
        # leaves of the size of the maximum radius as well.
        # The `DynamicVectorOfVectors` storage needs the same capacity for all lists,
        # so the tree needs the same capacity as the grid.
        fine_spacing = min_radius / search_radius_factor
        max_points_per_cell = ceil(Int, 2 * (max_radius / fine_spacing + 1)^NDIMS)

        tree_nhs = [TreeNeighborhoodSearch{NDIMS}(; n_points, pruning, neighbor_criterion,
                                                  cell_list = TreeCellList(; min_corner,
                                                                           max_corner,
                                                                           max_level,
                                                                           root_cell_size,
                                                                           max_points_per_cell,
                                                                           backend),
                                                  update_strategy = ParallelUpdate())
                    for (pruning, backend) in
                        ((false, DynamicVectorOfVectors{Int32}),
                         (true, DynamicVectorOfVectors{Int32}),
                         (true, CompactVectorOfVectors{Int32}))]

        grid_cell_list = FullGridCellList(; search_radius = max_radius, min_corner,
                                          max_corner,
                                          max_points_per_cell)
        grid_nhs = GridNeighborhoodSearch{NDIMS}(; search_radius = max_radius,
                                                 cell_list = grid_cell_list,
                                                 update_strategy = ParallelUpdate())

        neighborhood_searches = (tree_nhs..., grid_nhs)

        for i in eachindex(neighborhood_searches)
            neighborhood_search = PointNeighbors.Adapt.adapt(parallelization_backend,
                                                             neighborhood_searches[i])
            coords = PointNeighbors.Adapt.adapt(parallelization_backend, coordinates)
            radii = PointNeighbors.Adapt.adapt(parallelization_backend, search_radii)
            initialize_variable_radius!(neighborhood_search, coords, radii;
                                        parallelization_backend)

            time = benchmark(neighborhood_search, coords, radii; parallelization_backend,
                             neighbor_criterion)
            times[iter, i] = time
            time_string = BenchmarkTools.prettytime(time * 1e9)
            time_string_per_particle = BenchmarkTools.prettytime(time * 1e9 / n_points)
            println("$(names[i])")
            println("with $n_points particles " *
                    "finished in $time_string ($time_string_per_particle per particle)\n")
        end
    end

    return n_particles_vec, times
end
