using PointNeighbors
using TrixiParticles
using BenchmarkTools

# Create a dummy semidiscretization type to be able to use a specific neighborhood search
struct DummySemidiscretization{N, P}
    neighborhood_search     :: N
    parallelization_backend :: P
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

"""
    benchmark_wcsph(neighborhood_search, coordinates;
                    parallelization_backend = default_backend(coordinates))

A benchmark of the right-hand side of a full real-life Weakly Compressible
Smoothed Particle Hydrodynamics (WCSPH) simulation with TrixiParticles.jl.
This method is used to simulate an incompressible fluid.
"""
function benchmark_wcsph(neighborhood_search, coordinates;
                         parallelization_backend = default_backend(coordinates))
    density = 1000.0
    particle_spacing = PointNeighbors.search_radius(neighborhood_search) / 3
    fluid = InitialCondition(; coordinates, density, mass = 0.1, particle_spacing)

    sound_speed = 10.0
    state_equation = StateEquationCole(; sound_speed, reference_density = density,
                                       exponent = 1)

    viscosity = ArtificialViscosityMonaghan(alpha = 0.02, beta = 0.0)
    density_diffusion = DensityDiffusionMolteniColagrossi(delta = 0.1)

    __benchmark_wcsph_inner(neighborhood_search, fluid, state_equation,
                            viscosity, density_diffusion, parallelization_backend)
end

"""
    benchmark_wcsph_fp32(neighborhood_search, coordinates;
                         parallelization_backend = default_backend(coordinates))

Like [`benchmark_wcsph`](@ref), but using single precision floating point numbers.
"""
function benchmark_wcsph_fp32(neighborhood_search, coordinates_;
                              parallelization_backend = default_backend(coordinates_))
    coordinates = convert(Matrix{Float32}, coordinates_)
    density = 1000.0f0
    particle_spacing = PointNeighbors.search_radius(neighborhood_search) / 3
    fluid = InitialCondition(; coordinates, density, mass = 0.1f0, particle_spacing)

    sound_speed = 10.0f0
    state_equation = StateEquationCole(; sound_speed, reference_density = density,
                                       exponent = 1)

    viscosity = ArtificialViscosityMonaghan(alpha = 0.02f0, beta = 0.0f0)
    density_diffusion = DensityDiffusionMolteniColagrossi(delta = 0.1f0)

    __benchmark_wcsph_inner(neighborhood_search, fluid, state_equation,
                            viscosity, density_diffusion, parallelization_backend)
end

function __benchmark_wcsph_inner(neighborhood_search, initial_condition, state_equation,
                                 viscosity, density_diffusion, parallelization_backend)
    # Compact support == 2 * smoothing length for these kernels
    smoothing_length = PointNeighbors.search_radius(neighborhood_search) / 2
    if ndims(neighborhood_search) == 1
        smoothing_kernel = SchoenbergCubicSplineKernel{1}()
    else
        smoothing_kernel = WendlandC2Kernel{ndims(neighborhood_search)}()
    end

    fluid_system = WeaklyCompressibleSPHSystem(initial_condition, ContinuityDensity(),
                                               state_equation, smoothing_kernel,
                                               smoothing_length, viscosity = viscosity,
                                               density_diffusion = density_diffusion)

    system = PointNeighbors.Adapt.adapt(parallelization_backend, fluid_system)
    nhs = PointNeighbors.Adapt.adapt(parallelization_backend, neighborhood_search)
    semi = DummySemidiscretization(nhs, parallelization_backend)

    v = PointNeighbors.Adapt.adapt(parallelization_backend,
                                   vcat(initial_condition.velocity,
                                        initial_condition.density'))
    u = PointNeighbors.Adapt.adapt(parallelization_backend, initial_condition.coordinates)
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
    material = (density = 1000.0, E = 1.4e6, nu = 0.4)
    solid = InitialCondition(; coordinates, density = material.density, mass = 0.1)

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
    semi = DummySemidiscretization(neighborhood_search, parallelization_backend)

    v = copy(solid.velocity)
    u = copy(solid.coordinates)
    dv = zero(v)

    # Initialize the system
    TrixiParticles.initialize!(solid_system, semi)

    return @belapsed TrixiParticles.interact!($dv, $v, $u, $v, $u,
                                              $solid_system, $solid_system, $semi)
end
