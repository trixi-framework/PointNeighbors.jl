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

    # Note that we cannot just disable parallelism in TrixiParticles.
    # But passing a different backend like `CUDA.CUDABackend`
    # allows us to change the type of the array to run the benchmark on the GPU.
    if parallel isa Bool
        system = fluid_system
        nhs = neighborhood_search
    else
        system = PointNeighbors.Adapt.adapt(parallel, fluid_system)
        nhs = PointNeighbors.Adapt.adapt(parallel, neighborhood_search)
    end

    v = PointNeighbors.Adapt.adapt(parallel, vcat(fluid.velocity, fluid.density'))
    u = PointNeighbors.Adapt.adapt(parallel, coordinates)
    dv = zero(v)

    # Initialize the system
    TrixiParticles.initialize!(system, nhs)
    # Note that the third argument is supposed to be the semidiscretization, but it is only
    # used for `@threaded`, so we can just pass `v` instead.
    TrixiParticles.compute_pressure!(system, v, v)

    return @belapsed TrixiParticles.interact!($dv, $v, $u, $v, $u, $nhs, $system, $system)
end

"""
    benchmark_wcsph_fp32(neighborhood_search, coordinates; parallel = true)

Like [`benchmark_wcsph`](@ref), but using single precision floating point numbers.
"""
function benchmark_wcsph_fp32(neighborhood_search, coordinates_; parallel = true)
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
                                               acceleration = (0.0f0, 0.0f0, 0.0f0),
                                               density_diffusion = density_diffusion)

    # Note that we cannot just disable parallelism in TrixiParticles.
    # But passing a different backend like `CUDA.CUDABackend`
    # allows us to change the type of the array to run the benchmark on the GPU.
    if parallel isa Bool
        system = fluid_system
        nhs = neighborhood_search
    else
        system = PointNeighbors.Adapt.adapt(parallel, fluid_system)
        nhs = PointNeighbors.Adapt.adapt(parallel, neighborhood_search)
    end

    v = PointNeighbors.Adapt.adapt(parallel, vcat(fluid.velocity, fluid.density'))
    u = PointNeighbors.Adapt.adapt(parallel, coordinates)
    dv = zero(v)

    # Initialize the system
    TrixiParticles.initialize!(system, nhs)
    TrixiParticles.compute_pressure!(system, v)

    return @belapsed TrixiParticles.interact!($dv, $v, $u, $v, $u, $nhs, $system, $system)
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

    # Compact support == 2 * smoothing length for these kernels
    smoothing_length = PointNeighbors.search_radius(neighborhood_search) / 2
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
