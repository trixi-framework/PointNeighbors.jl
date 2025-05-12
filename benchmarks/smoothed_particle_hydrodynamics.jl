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
    benchmark_wcsph(neighborhood_search, coordinates; parallel = true)

A benchmark of the right-hand side of a full real-life Weakly Compressible
Smoothed Particle Hydrodynamics (WCSPH) simulation with TrixiParticles.jl.
This method is used to simulate an incompressible fluid.
"""
function benchmark_wcsph(neighborhood_search, coordinates;
                         parallelization_backend = default_backend(coordinates))
    density = 1000.0
    fluid = InitialCondition(; coordinates, density, mass = 0.1)

    # Compact support == 2 * smoothing length for these kernels
    smoothing_length = PointNeighbors.search_radius(neighborhood_search) / 2
    if ndims(neighborhood_search) == 1
        smoothing_kernel = SchoenbergCubicSplineKernel{1}()
    else
        smoothing_kernel = WendlandC2Kernel{ndims(neighborhood_search)}()
    end

    sound_speed = 10.0
    state_equation = StateEquationCole(; sound_speed, reference_density = density,
                                       exponent = 1)

    fluid_density_calculator = ContinuityDensity()
    viscosity = ArtificialViscosityMonaghan(alpha = 0.02, beta = 0.0)
    density_diffusion = DensityDiffusionMolteniColagrossi(delta = 0.1)

    fluid_system = WeaklyCompressibleSPHSystem(fluid, fluid_density_calculator,
                                               state_equation, smoothing_kernel,
                                               smoothing_length, viscosity = viscosity,
                                               density_diffusion = density_diffusion)

    system = PointNeighbors.Adapt.adapt(parallelization_backend, fluid_system)
    nhs = PointNeighbors.Adapt.adapt(parallelization_backend, neighborhood_search)
    semi = DummySemidiscretization(nhs, parallelization_backend)

    v = PointNeighbors.Adapt.adapt(parallelization_backend,
                                   vcat(fluid.velocity, fluid.density'))
    u = PointNeighbors.Adapt.adapt(parallelization_backend, coordinates)
    dv = zero(v)

    # Initialize the system
    TrixiParticles.initialize!(system, semi)
    TrixiParticles.compute_pressure!(system, v, semi)

    return @belapsed TrixiParticles.interact!($dv, $v, $u, $v, $u, $system, $system, $semi)
end

"""
    benchmark_wcsph_fp32(neighborhood_search, coordinates; parallel = true)

Like [`benchmark_wcsph`](@ref), but using single precision floating point numbers.
"""
function benchmark_wcsph_fp32(neighborhood_search, coordinates_;
                              parallelization_backend = default_backend(coordinates_))
    coordinates = convert(Matrix{Float32}, coordinates_)
    density = 1000.0f0
    fluid = InitialCondition(; coordinates, density, mass = 0.1f0)

    # Compact support == 2 * smoothing length for these kernels
    smoothing_length = convert(Float32,
                               PointNeighbors.search_radius(neighborhood_search) / 2)
    if ndims(neighborhood_search) == 1
        smoothing_kernel = SchoenbergCubicSplineKernel{1}()
    else
        smoothing_kernel = WendlandC2Kernel{ndims(neighborhood_search)}()
    end

    sound_speed = 10.0f0
    state_equation = StateEquationCole(; sound_speed, reference_density = density,
                                       exponent = 1)

    fluid_density_calculator = ContinuityDensity()
    viscosity = ArtificialViscosityMonaghan(alpha = 0.02f0, beta = 0.0f0)
    density_diffusion = DensityDiffusionMolteniColagrossi(delta = 0.1f0)

    fluid_system = WeaklyCompressibleSPHSystem(fluid, fluid_density_calculator,
                                               state_equation, smoothing_kernel,
                                               smoothing_length, viscosity = viscosity,
                                               acceleration = ntuple(_ -> 0.0f0,
                                                                     Val(ndims(neighborhood_search))),
                                               density_diffusion = density_diffusion)

    system = PointNeighbors.Adapt.adapt(parallelization_backend, fluid_system)
    nhs = PointNeighbors.Adapt.adapt(parallelization_backend, neighborhood_search)
    semi = DummySemidiscretization(nhs, parallelization_backend)

    v = PointNeighbors.Adapt.adapt(parallelization_backend,
                                   vcat(fluid.velocity, fluid.density'))
    u = PointNeighbors.Adapt.adapt(parallelization_backend, coordinates)
    dv = zero(v)

    # Initialize the system
    TrixiParticles.initialize!(system, semi)
    TrixiParticles.compute_pressure!(system, v, semi)

    return @belapsed TrixiParticles.interact!($dv, $v, $u, $v, $u, $system, $system, $semi)
end

"""
    benchmark_tlsph(neighborhood_search, coordinates; parallel = true)

A benchmark of the right-hand side of a full real-life Total Lagrangian
Smoothed Particle Hydrodynamics (TLSPH) simulation with TrixiParticles.jl.
This method is used to simulate an elastic structure.
"""
function benchmark_tlsph(neighborhood_search, coordinates;
                         parallelization_backend = default_backend(coordinates))
    material = (density = 1000.0, E = 1.4e6, nu = 0.4)
    solid = InitialCondition(; coordinates, density = material.density, mass = 0.1)

    # Compact support == 2 * smoothing length for these kernels
    smoothing_length = PointNeighbors.search_radius(neighborhood_search) / 2
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
