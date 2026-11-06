using PointNeighbors
using BenchmarkTools

# For `perturb!`
include("../test/point_cloud.jl")

"""
    benchmark_initialize(neighborhood_search, coordinates;
                         parallelization_backend = default_backend(coordinates))

Benchmark neighborhood search initialization with the given `coordinates`.
"""
function benchmark_initialize(neighborhood_search, coordinates;
                              parallelization_backend = default_backend(coordinates))
    return @belapsed $initialize!($neighborhood_search, $coordinates, $coordinates;
                                  parallelization_backend = $parallelization_backend)
end

"""
    benchmark_update_alternating(neighborhood_search, coordinates;
                                 parallelization_backend = default_backend(coordinates))

A very simple benchmark for neighborhood search update, alternating between two differently
perturbed point clouds.

This is a good benchmark for incremental updates, since most particles stay in their cells.
Similar to 3D dam break simulations, ~0.5% of the particles change their cell during
an update in 2D and ~0.7% in 3D.
"""
function benchmark_update_alternating(neighborhood_search, coordinates;
                                      parallelization_backend = default_backend(coordinates))
    # `perturb!` only works on the CPU, so we perturb a copy on the CPU
    # and move it to the device afterwards.
    coordinates2 = copy(PointNeighbors.Adapt.adapt(Array, coordinates))
    # Perturb all coordinates with a perturbation factor of `4e-4`.
    # This factor was tuned so that ~0.5% of the particles change their cell during an
    # update in 2D and ~0.7% in 3D.
    # These values are the same as the experimentally computed averages in 2D and 3D SPH
    # dam break simulations. So this benchmark replicates a real-life SPH update.
    # Use the same perturbation every time to make comparisons fair and reproducible.
    rng = Random.Xoshiro(1)
    perturb!(coordinates2,
             4e-4 * PointNeighbors.search_radius(neighborhood_search); rng)
    coordinates2 = PointNeighbors.Adapt.adapt(parallelization_backend, coordinates2)

    function update_alternating!(neighborhood_search, coordinates, coordinates2,
                                 parallelization_backend)
        update!(neighborhood_search, coordinates, coordinates; parallelization_backend)
        update!(neighborhood_search, coordinates, coordinates2; parallelization_backend)
    end

    result = @belapsed $update_alternating!($neighborhood_search, $coordinates,
                                            $coordinates2, $parallelization_backend)

    # Return average update time
    return 0.5 * result
end

"""
    benchmark_sorting(neighborhood_search, coordinates;
                      parallelization_backend = default_backend(coordinates))

Benchmark sorting the particles of a simulation by cell with a neighborhood search that
sorts the points in every update (see `PointNeighbors.sorts_points`), like
a `GridNeighborhoodSearch` with a `FullGridCellList` with the `CompactVectorOfVectors`
backend (see the keyword `sort_points` of [`run_benchmark_full_grid`](@ref)).

Each benchmark iteration does all the work that TrixiParticles.jl does for sorting
the particles in every update, except reordering the arrays of the systems,
which is negligible:
1. Update the neighborhood search with the coordinates in the previous sorted order.
2. Compose the permutation of the integration arrays with the new permutation.
3. Copy `v` and `u` from the integration arrays to arrays in the sorted order.
4. Copy `dv` from the sorted order back to the order of the integration array.

The particles are first sorted like [`benchmark_wcsph`](@ref) does, so that the runtime
of [`benchmark_wcsph`](@ref) with sorting minus the runtime of this benchmark is
the runtime of the interaction on the sorted particles.
`v` and `dv` have `ndims + 1` variables per particle like WCSPH with `ContinuityDensity`.
"""
function benchmark_sorting(neighborhood_search, coordinates;
                           parallelization_backend = default_backend(coordinates))
    if !PointNeighbors.sorts_points(query_search(neighborhood_search))
        throw(ArgumentError("`benchmark_sorting` requires a neighborhood search " *
                            "that sorts the points"))
    end

    # The integration arrays like in `benchmark_wcsph`. The values of `v` don't matter.
    u = coordinates
    v = similar(u, size(u, 1) + 1, size(u, 2))
    v .= 1
    dv = zero(v)

    sorting = BenchmarkParticleSorting(v, u, dv, parallelization_backend)

    # The neighborhood search has been initialized with the unsorted particles by
    # `run_benchmark`. Sort the particles for the first time.
    update_permutation!(sorting, neighborhood_search, parallelization_backend)

    function sort!(dv, v, u, sorting, neighborhood_search, parallelization_backend)
        update_and_sort!(sorting, neighborhood_search, u, parallelization_backend)
        copy_to_sorted!(sorting, v, u, parallelization_backend)
        copy_from_sorted!(dv, sorting, parallelization_backend)

        return dv
    end

    return @belapsed $sort!($dv, $v, $u, $sorting, $neighborhood_search,
                            $parallelization_backend)
end

# Sorting particles like TrixiParticles.jl (see `src/general/particle_sorting.jl` in
# TrixiParticles.jl). The integration arrays `v`, `u` and `dv` are never reordered.
# Instead, `v` and `u` are copied to arrays in the sorted order (the system order),
# and `dv` is copied back.
struct BenchmarkParticleSorting{I, V, U, DV}
    # `original_index[i]` is the index in the integration arrays of the particle with
    # index `i` in the system order. `sorted_index` is the inverse permutation.
    original_index :: I
    sorted_index   :: I
    index_buffer   :: I
    # Integration arrays in system order
    v_sorted       :: V
    u_sorted       :: U
    dv_sorted      :: DV
end

function BenchmarkParticleSorting(v, u, dv, parallelization_backend)
    original_index = PointNeighbors.Adapt.adapt(parallelization_backend, collect(axes(u, 2)))
    dv_sorted = sorted_array(dv, parallelization_backend)
    dv_sorted .= 0

    return BenchmarkParticleSorting(original_index, similar(original_index),
                                    similar(original_index),
                                    sorted_array(v, parallelization_backend),
                                    sorted_array(u, parallelization_backend), dv_sorted)
end

# An array for the integration array `array` in system order.
# TrixiParticles.jl might use a different layout for these arrays.
sorted_array(array, parallelization_backend) = similar(array)

# Update the neighborhood search with the coordinates in the previous system order,
# which `wrap_u` of TrixiParticles.jl returns as a view of the integration array,
# and compose the permutation (see `update_nhs_and_sort!` in TrixiParticles.jl).
# Return the new permutation of the system order.
function update_and_sort!(sorting, neighborhood_search, u, parallelization_backend)
    u_system = view(u, :, sorting.original_index)
    benchmark_update!(neighborhood_search, u_system; parallelization_backend)

    return update_permutation!(sorting, neighborhood_search, parallelization_backend)
end

# Update `original_index` and `sorted_index` with the permutation of the neighborhood
# search, which has been updated with the coordinates in the previous system order
# (see `sort_particles_by_nhs!` in TrixiParticles.jl).
# Return the new permutation of the system order.
function update_permutation!(sorting, neighborhood_search, parallelization_backend)
    (; original_index, sorted_index, index_buffer) = sorting

    # `old_index[i]` is the index in the previous system order of the particle
    # that is now the `i`-th particle
    old_index = PointNeighbors.point_permutation(query_search(neighborhood_search))

    copy_parallel!(index_buffer, original_index, parallelization_backend)
    PointNeighbors.@threaded parallelization_backend for i in eachindex(original_index)
        @inbounds original_index[i] = index_buffer[old_index[i]]
    end

    PointNeighbors.@threaded parallelization_backend for i in eachindex(original_index)
        @inbounds sorted_index[original_index[i]] = i
    end

    return old_index
end

# Copy `v` and `u` to arrays in system order (see `sorted_ode_arrays!` in TrixiParticles.jl)
function copy_to_sorted!(sorting, v, u, parallelization_backend)
    copy_permuted!(sorting.v_sorted, v, sorting.original_index, parallelization_backend)
    copy_permuted!(sorting.u_sorted, u, sorting.original_index, parallelization_backend)

    return sorting
end

# `dest[:, i] = src[:, permutation[i]]` for all `i`
function copy_permuted!(dest, src, permutation, parallelization_backend)
    PointNeighbors.@threaded parallelization_backend for i in axes(dest, 2)
        particle = @inbounds permutation[i]
        for j in axes(dest, 1)
            @inbounds dest[j, i] = src[j, particle]
        end
    end

    return dest
end

# Copy `dv_sorted` from system order back to the order of the integration array `dv`
# (see `unsort_dv!` in TrixiParticles.jl).
# Gathering from the sorted array is faster than scattering to `dv` on GPUs.
function copy_from_sorted!(dv, sorting, parallelization_backend)
    (; dv_sorted, sorted_index) = sorting

    PointNeighbors.@threaded parallelization_backend for particle in axes(dv, 2)
        for i in axes(dv, 1)
            @inbounds dv[i, particle] = dv_sorted[i, sorted_index[particle]]
        end
    end

    return dv
end

# `dest .= src`, but in parallel
function copy_parallel!(dest, src, parallelization_backend)
    PointNeighbors.@threaded parallelization_backend for i in eachindex(dest, src)
        @inbounds dest[i] = src[i]
    end

    return dest
end
