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

    # Perturb the initial velocity so that approaching particle pairs exercise the
    # `vr < 0` branch of ArtificialViscosityMonaghan.
    fluid.velocity .+= convert(ELTYPE, 1.0e-3) .* randn(ELTYPE, size(fluid.velocity))

    # Make sure that the computed forces are not all zero
    for i in eachindex(fluid.density)
        fluid.density[i] += randn(eltype(fluid.density))
    end

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

    fluid_system = WeaklyCompressibleSPHSystem(fluid;
                                               density_calculator = ContinuityDensity(),
                                               state_equation, smoothing_kernel,
                                               smoothing_length, viscosity,
                                               density_diffusion)

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
