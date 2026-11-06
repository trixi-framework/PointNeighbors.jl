using PointNeighbors
using PointNeighbors.Adapt
using TrixiParticles
using BenchmarkTools
using Random

# Create a dummy semidiscretization type to be able to use a specific neighborhood search
struct DummySemidiscretization{N, P, IT}
    neighborhood_search     :: N
    parallelization_backend :: P
    integrate_tlsph         :: IT
end

@inline function PointNeighbors.parallel_foreach(f, iterator, semi::DummySemidiscretization)
    PointNeighbors.parallel_foreach(f, iterator, semi.parallelization_backend)
end

@inline function TrixiParticles.get_neighborhood_search(_, _, semi::DummySemidiscretization)
    return semi.neighborhood_search
end

@inline function TrixiParticles.get_neighborhood_search(_, semi::DummySemidiscretization)
    return semi.neighborhood_search
end

# Newer TrixiParticles versions define TLSPH-specific neighborhood-search lookups.
@inline function TrixiParticles.get_neighborhood_search(::TotalLagrangianSPHSystem,
                                                        semi::DummySemidiscretization)
    return semi.neighborhood_search
end

@inline function TrixiParticles.get_neighborhood_search(::TotalLagrangianSPHSystem,
                                                        ::TotalLagrangianSPHSystem,
                                                        semi::DummySemidiscretization)
    return semi.neighborhood_search
end

"""
    benchmark_wcsph(neighborhood_search, coordinates;
                    parallelization_backend = default_backend(coordinates))

A benchmark of the right-hand side of a full real-life Weakly Compressible
Smoothed Particle Hydrodynamics (WCSPH) simulation with TrixiParticles.jl.
This method is used to simulate an incompressible fluid.

When the neighborhood search sorts the points in every update
(see `PointNeighbors.sorts_points`), each benchmark iteration first sorts the particles
like `kick!` does when TrixiParticles.jl sorts the particles in every update,
so that the benchmark includes the sorting.
See [`benchmark_sorting`](@ref) for a benchmark of only the sorting.
"""
function benchmark_wcsph(neighborhood_search, coordinates;
                         parallelization_backend = default_backend(coordinates))
    # System initialization has to happen on the CPU
    coordinates_cpu = PointNeighbors.Adapt.adapt(Array, coordinates)

    # The neighborhood search to find the neighbors (see `query_search`)
    nhs_query = query_search(neighborhood_search)

    search_radius = PointNeighbors.search_radius(nhs_query)
    ELTYPE = typeof(search_radius)
    density = convert(ELTYPE, 1000.0)
    particle_spacing = search_radius / 3
    fluid = InitialCondition(; coordinates = coordinates_cpu, density,
                             mass = convert(ELTYPE, 0.1) * particle_spacing,
                             particle_spacing)

    # Use the same perturbation every time to make comparisons fair and reproducible.
    rng = Random.Xoshiro(1)

    # Perturb the initial velocity so that approaching particle pairs exercise the
    # `vr < 0` branch of ArtificialViscosityMonaghan.
    fluid.velocity .+= convert(ELTYPE, 1.0e-3) .* randn(rng, ELTYPE, size(fluid.velocity))

    # Make sure that the computed forces are not all zero
    for i in eachindex(fluid.density)
        fluid.density[i] += randn(rng, eltype(fluid.density))
    end

    sound_speed = convert(ELTYPE, 10.0)
    state_equation = StateEquationCole(; sound_speed, reference_density = density,
                                       exponent = 1)

    viscosity = ArtificialViscosityMonaghan(alpha = convert(ELTYPE, 0.02),
                                            beta = convert(ELTYPE, 0.0))
    density_diffusion = DensityDiffusionMolteniColagrossi(delta = convert(ELTYPE, 0.1))

    # Compact support == 2 * smoothing length for these kernels
    smoothing_length = search_radius / 2
    if ndims(nhs_query) == 1
        smoothing_kernel = SchoenbergCubicSplineKernel{1}()
    else
        smoothing_kernel = WendlandC2Kernel{ndims(nhs_query)}()
    end

    fluid_system = WeaklyCompressibleSPHSystem(fluid;
                                               density_calculator = ContinuityDensity(),
                                               state_equation, smoothing_kernel,
                                               smoothing_length, viscosity,
                                               density_diffusion)

    system = Adapt.adapt(parallelization_backend, fluid_system)

    v = Adapt.adapt(parallelization_backend,
                    vcat(fluid.velocity, fluid.density'))
    u = Adapt.adapt(parallelization_backend, fluid.coordinates)
    dv = zero(v)

    if PointNeighbors.sorts_points(nhs_query)
        return benchmark_wcsph_sorted(dv, v, u, system, neighborhood_search,
                                      parallelization_backend)
    end

    # Remove unnecessary data structures that are only used for initialization
    nhs = PointNeighbors.freeze_neighborhood_search(nhs_query)

    semi = DummySemidiscretization(nhs, parallelization_backend, true)

    # Initialize the system
    TrixiParticles.initialize!(system, semi)
    TrixiParticles.compute_pressure!(system, v, semi)

    return @belapsed TrixiParticles.interact!($dv, $v, $u, $v, $u, $system, $system, $semi)
end

# Benchmark the WCSPH interaction with a neighborhood search that sorts the points
# in every update. Each benchmark iteration first sorts the particles like `kick!` does
# when TrixiParticles.jl sorts the particles in every update
# (see `src/general/particle_sorting.jl` in TrixiParticles.jl):
# 1. Update the neighborhood search with the coordinates in the previous system order
#    and compose the permutation (`update_nhs_and_sort!`).
# 2. Reorder the arrays of the system with the new permutation (`sort_particles_by_nhs!`).
# 3. Copy `v` and `u` to arrays in system order (`sorted_ode_arrays!`).
# 4. Compute the interaction on the sorted particles.
# 5. Copy `dv` back to the order of the integration arrays (`unsort_dv!`).
# All steps except 2 and 4 are benchmarked by `benchmark_sorting`.
#
# The integration arrays `v`, `u` and `dv` are never reordered. Their order is given by
# `run_benchmark` and almost sorted (see `sort_displacement` in `run_benchmark`), like
# the integration arrays in a simulation that are only sorted by the `SortingCallback`.
function benchmark_wcsph_sorted(dv, v, u, system, neighborhood_search,
                                parallelization_backend)
    sorting = BenchmarkParticleSorting(v, u, dv, parallelization_backend)

    arrays = system_arrays_to_reorder(system)
    buffers = map(similar, arrays)

    # The neighborhood search has been initialized with the unsorted particles by
    # `run_benchmark`. Sort the particles for the first time like `create_particle_sorting`
    # does and compute the pressure in system order.
    old_index = update_permutation!(sorting, neighborhood_search, parallelization_backend)
    reorder_arrays!(arrays, buffers, old_index, parallelization_backend)

    semi = sorted_semi(neighborhood_search, parallelization_backend)
    TrixiParticles.initialize!(system, semi)
    copy_to_sorted!(sorting, v, u, parallelization_backend)
    TrixiParticles.compute_pressure!(system, sorting.v_sorted, semi)

    function sort_and_interact!(dv, v, u, system, neighborhood_search, sorting, arrays,
                                buffers, parallelization_backend)
        old_index = update_and_sort!(sorting, neighborhood_search, u,
                                     parallelization_backend)
        reorder_arrays!(arrays, buffers, old_index, parallelization_backend)
        copy_to_sorted!(sorting, v, u, parallelization_backend)

        (; dv_sorted, v_sorted, u_sorted) = sorting
        semi = sorted_semi(neighborhood_search, parallelization_backend)
        TrixiParticles.interact!(dv_sorted, v_sorted, u_sorted, v_sorted, u_sorted,
                                 system, system, semi)

        copy_from_sorted!(dv, sorting, parallelization_backend)

        return dv
    end

    return @belapsed $sort_and_interact!($dv, $v, $u, $system, $neighborhood_search,
                                         $sorting, $arrays, $buffers,
                                         $parallelization_backend)
end

# The semidiscretization for the interaction of the sorted particles.
# Like TrixiParticles.jl, we create the neighborhood search for the sorted particles
# in every update. This is cheap, since it shares its data with `neighborhood_search`.
function sorted_semi(neighborhood_search, parallelization_backend)
    sorted_nhs = PointNeighbors.sorted_neighborhood_search(query_search(neighborhood_search))

    return DummySemidiscretization(sorted_nhs, parallelization_backend, true)
end

# Arrays of `system` that have to be reordered when the particles are sorted.
# TrixiParticles.jl only reorders the mass (see `particle_arrays_to_sort`) and recomputes
# the pressure and the other per-particle quantities in every update after sorting.
# We reorder them instead, since the benchmark without sorting doesn't compute them either.
# Arrays that are not stored in memory (like a constant mass) are not reordered.
function system_arrays_to_reorder(system)
    n_particles = TrixiParticles.nparticles(system)
    cache_arrays = Tuple(array for array in values(system.cache)
                         if array isa AbstractVector && length(array) == n_particles)
    arrays = (system.mass, system.pressure, cache_arrays...)

    return Tuple(array for array in arrays if array isa DenseArray)
end

# `array[i] = array[old_index[i]]` for all arrays
function reorder_arrays!(arrays, buffers, old_index, parallelization_backend)
    foreach(arrays, buffers) do array, buffer
        copy_parallel!(buffer, array, parallelization_backend)
        PointNeighbors.@threaded parallelization_backend for i in eachindex(array)
            @inbounds array[i] = buffer[old_index[i]]
        end
    end

    return arrays
end

"""
    benchmark_tlsph(neighborhood_search, coordinates;
                    parallelization_backend = default_backend(coordinates))

A benchmark of the interaction forces of a full real-life Total Lagrangian
Smoothed Particle Hydrodynamics (TLSPH) simulation with TrixiParticles.jl.
This method is used to simulate an elastic structure.

The right-hand side of the TLSPH equations consists of two main parts:
- The deformation gradient ([`benchmark_tlsph_deformation_grad`](@ref)).
- The interaction forces ([`benchmark_tlsph`](@ref)).
"""
function benchmark_tlsph(neighborhood_search, coordinates;
                         parallelization_backend = default_backend(coordinates))
    (dv, v, system,
     semi) = setup_tlsph(neighborhood_search, coordinates, parallelization_backend)

    return @belapsed TrixiParticles.interact_structure_structure!($dv, $v, $system, $semi)
end

"""
    benchmark_tlsph_deformation_grad(neighborhood_search, coordinates;
                                     parallelization_backend = default_backend(coordinates))

A benchmark of the deformation gradient computation of a full real-life Total Lagrangian
Smoothed Particle Hydrodynamics (TLSPH) simulation with TrixiParticles.jl.
This method is used to simulate an elastic structure.

The right-hand side of the TLSPH equations consists of two main parts:
- The deformation gradient ([`benchmark_tlsph_deformation_grad`](@ref)).
- The interaction forces ([`benchmark_tlsph`](@ref)).
"""
function benchmark_tlsph_deformation_grad(neighborhood_search, coordinates;
                                          parallelization_backend = default_backend(coordinates))
    (dv, v, system,
     semi) = setup_tlsph(neighborhood_search, coordinates, parallelization_backend)
    deformation_grad = system.deformation_grad

    return @belapsed TrixiParticles.calc_deformation_grad!($deformation_grad, $system,
                                                           $semi)
end

function setup_tlsph(neighborhood_search, coordinates, parallelization_backend)
    # System initialization has to happen on the CPU
    coordinates_cpu = PointNeighbors.Adapt.adapt(Array, coordinates)

    search_radius = PointNeighbors.search_radius(neighborhood_search)
    ELTYPE = typeof(search_radius)
    material = (density = convert(ELTYPE, 1000.0), E = convert(ELTYPE, 1.4e6),
                nu = convert(ELTYPE, 0.4))

    # The `particle_spacing` is only required for setting the type of the initial condition
    solid = InitialCondition(; coordinates = coordinates_cpu,
                             density = material.density, mass = convert(ELTYPE, 0.1),
                             particle_spacing = search_radius)

    # Compact support == 2 * smoothing length for these kernels
    smoothing_length_ = PointNeighbors.search_radius(neighborhood_search) / 2
    smoothing_length = convert(typeof(material.E), smoothing_length_)
    if ndims(neighborhood_search) == 1
        smoothing_kernel = SchoenbergCubicSplineKernel{1}()
    else
        smoothing_kernel = WendlandC2Kernel{ndims(neighborhood_search)}()
    end

    penalty_force = PenaltyForceGanzenmueller(alpha = convert(ELTYPE, 0.1))
    solid_system = TotalLagrangianSPHSystem(solid; smoothing_kernel, smoothing_length,
                                            young_modulus = material.E,
                                            poisson_ratio = material.nu, penalty_force)
    system_ = Adapt.adapt(parallelization_backend, solid_system)

    # Remove unnecessary data structures that are only used for initialization
    nhs = PointNeighbors.freeze_neighborhood_search(neighborhood_search)
    system = TrixiParticles.@set system_.self_interaction_nhs = nhs

    semi = DummySemidiscretization(nhs, parallelization_backend, true)

    v = Adapt.adapt(parallelization_backend, copy(solid.velocity))
    dv = zero(v)

    # Initialize the system
    TrixiParticles.initialize!(system, semi)
    TrixiParticles.compute_pk1_corrected!(system, semi)

    return dv, v, system, semi
end
