using PointNeighbors
using TrixiParticles
using BenchmarkTools

"""
    benchmark_wcsph(neighborhood_search, coordinates; parallel = true)

A benchmark of the right-hand side of a full real-life Weakly Compressible
Smoothed Particle Hydrodynamics (WCSPH) simulation with TrixiParticles.jl.
This method is used to simulate an incompressible fluid.
"""
function benchmark_wcsph(neighborhood_search, coordinates; parallel = true)
    density = 1000.0
    fluid = InitialCondition(; coordinates, density, mass = 0.1)

    # Compact support == smoothing length for the Wendland kernel
    smoothing_length = PointNeighbors.search_radius(neighborhood_search)
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

    v = vcat(fluid.velocity, fluid.density')
    u = copy(fluid.coordinates)
    dv = zero(v)

    # Initialize the system
    TrixiParticles.initialize!(fluid_system, neighborhood_search)
    TrixiParticles.compute_pressure!(fluid_system, v)

    return @belapsed TrixiParticles.interact!($dv, $v, $u, $v, $u, $neighborhood_search,
                                              $fluid_system, $fluid_system)
end

"""
    benchmark_tlsph(neighborhood_search, coordinates; parallel = true)

A benchmark of the right-hand side of a full real-life Total Lagrangian
Smoothed Particle Hydrodynamics (TLSPH) simulation with TrixiParticles.jl.
This method is used to simulate an elastic structure.
"""
function benchmark_tlsph(neighborhood_search, coordinates; parallel = true)
    material = (density = 1000.0, E = 1.4e6, nu = 0.4)
    solid = InitialCondition(; coordinates, density = material.density, mass = 0.1)

    # Compact support == smoothing length for the Wendland kernel
    smoothing_length = PointNeighbors.search_radius(neighborhood_search)
    if ndims(neighborhood_search) == 1
        smoothing_kernel = SchoenbergCubicSplineKernel{1}()
    else
        smoothing_kernel = WendlandC2Kernel{ndims(neighborhood_search)}()
    end

    solid_system = TotalLagrangianSPHSystem(solid, smoothing_kernel, smoothing_length,
                                            material.E, material.nu)

    v = copy(solid.velocity)
    u = copy(solid.coordinates)
    dv = zero(v)

    # Initialize the system
    TrixiParticles.initialize!(solid_system, neighborhood_search)

    return @belapsed TrixiParticles.interact!($dv, $v, $u, $v, $u, $neighborhood_search,
                                              $solid_system, $solid_system)
end
