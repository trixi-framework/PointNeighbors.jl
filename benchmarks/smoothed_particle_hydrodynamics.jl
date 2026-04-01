using PointNeighbors
using PointNeighbors.Adapt
using TrixiParticles
using BenchmarkTools

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
"""
function benchmark_wcsph(neighborhood_search, coordinates;
                         parallelization_backend = default_backend(coordinates))
    # System initialization has to happen on the CPU
    coordinates_cpu = PointNeighbors.Adapt.adapt(Array, coordinates)

    search_radius = PointNeighbors.search_radius(neighborhood_search)
    ELTYPE = typeof(search_radius)
    density = convert(ELTYPE, 1000.0)
    particle_spacing = PointNeighbors.search_radius(neighborhood_search) / 3
    fluid = InitialCondition(; coordinates = coordinates_cpu, density,
                             mass = convert(ELTYPE, 0.1) * particle_spacing,
                             particle_spacing)

    sound_speed = convert(ELTYPE, 10.0)
    state_equation = StateEquationCole(; sound_speed, reference_density = density,
                                       exponent = 1)

    viscosity = ArtificialViscosityMonaghan(alpha = convert(ELTYPE, 0.02),
                                            beta = convert(ELTYPE, 0.0))
    density_diffusion = DensityDiffusionMolteniColagrossi(delta = convert(ELTYPE, 0.1))

    # Compact support == 2 * smoothing length for these kernels
    smoothing_length = PointNeighbors.search_radius(neighborhood_search) / 2
    if ndims(neighborhood_search) == 1
        smoothing_kernel = SchoenbergCubicSplineKernel{1}()
    else
        smoothing_kernel = WendlandC2Kernel{ndims(neighborhood_search)}()
    end

    fluid_system = WeaklyCompressibleSPHSystem(fluid, ContinuityDensity(),
                                               state_equation, smoothing_kernel,
                                               smoothing_length, viscosity = viscosity,
                                               density_diffusion = density_diffusion)

    system = Adapt.adapt(parallelization_backend, fluid_system)

    # Remove unnecessary data structures that are only used for initialization
    nhs = PointNeighbors.freeze_neighborhood_search(neighborhood_search)

    semi = DummySemidiscretization(nhs, parallelization_backend, true)

    v = Adapt.adapt(parallelization_backend,
                    vcat(fluid.velocity, fluid.density'))
    u = Adapt.adapt(parallelization_backend, fluid.coordinates)
    dv = zero(v)

    # Initialize the system
    TrixiParticles.initialize!(system, semi)
    TrixiParticles.compute_pressure!(system, v, semi)

    return @belapsed TrixiParticles.interact!($dv, $v, $u, $v, $u, $system, $system, $semi)
end

"""
    benchmark_tlsph(neighborhood_search, coordinates;
                    parallelization_backend = default_backend(coordinates))

A benchmark of the right-hand side of a full real-life Total Lagrangian
Smoothed Particle Hydrodynamics (TLSPH) simulation with TrixiParticles.jl.
This method is used to simulate an elastic structure.
"""
function benchmark_tlsph(neighborhood_search, coordinates;
                         parallelization_backend = default_backend(coordinates))
    # System initialization has to happen on the CPU
    coordinates_cpu = PointNeighbors.Adapt.adapt(Array, coordinates)

    search_radius = PointNeighbors.search_radius(neighborhood_search)
    ELTYPE = typeof(search_radius)
    material = (density = convert(ELTYPE, 1000.0), E = convert(ELTYPE, 1.4e6),
                nu = convert(ELTYPE, 0.4))
    solid = InitialCondition(; coordinates = coordinates_cpu,
                             density = material.density, mass = convert(ELTYPE, 0.1))

    # Compact support == 2 * smoothing length for these kernels
    smoothing_length_ = PointNeighbors.search_radius(neighborhood_search) / 2
    smoothing_length = convert(typeof(material.E), smoothing_length_)
    if ndims(neighborhood_search) == 1
        smoothing_kernel = SchoenbergCubicSplineKernel{1}()
    else
        smoothing_kernel = WendlandC2Kernel{ndims(neighborhood_search)}()
    end

    solid_system = TotalLagrangianSPHSystem(solid, smoothing_kernel, smoothing_length,
                                            material.E, material.nu)
    system_ = Adapt.adapt(parallelization_backend, solid_system)

    # Remove unnecessary data structures that are only used for initialization
    nhs = PointNeighbors.freeze_neighborhood_search(neighborhood_search)
    system = TrixiParticles.@set system_.self_interaction_nhs = nhs

    semi = DummySemidiscretization(nhs, parallelization_backend, true)

    v = Adapt.adapt(parallelization_backend, copy(solid.velocity))
    u = Adapt.adapt(parallelization_backend, copy(solid.coordinates))
    dv = zero(v)

    # Initialize the system
    TrixiParticles.initialize!(system, semi)

    return @belapsed TrixiParticles.interact_structure_structure2!($dv, $v, $system, $semi)
end
