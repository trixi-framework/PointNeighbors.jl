# Run all these benchmarks like this. The type of `search_radius_factor` determines
# if the benchmark is run in Float32 or Float64.
#
# include(joinpath(pkgdir(PointNeighbors),  "benchmarks", "benchmarks.jl"));
# _, times = run_benchmark_full_grid(benchmark_wcsph, (10, 10, 10), 9,
#                                    search_radius_factor=3.0f0,
#                                    parallelization_backend=backend);
#
# The machines listed below are the following:
# - Nvidia RTX 5090
# - Nvidia H100
# - AMD Instinct MI300A
# - 2x AMD EPYC 9965 x 192 (2 sockets, 384 cores total)
include("benchmarks.jl")

n_particles_vec = [1000, 4096, 15625, 64000, 250047, 1030301, 4096000, 16387064, 65450827]

# _, times = run_benchmark_full_grid(benchmark_wcsph, ...)
times_h100_fp64 = [0.000337699; 0.000401732; 0.000480261; 0.000522277; 0.001476047; 0.006047292; 0.023739696; 0.099076906; 0.397741399;;]
times_h100_fp32 = [0.00025341; 0.000311011; 0.000368099; 0.000390308; 0.001002314; 0.003407714; 0.013074915; 0.052667697; 0.212927902;;]
times_5090_fp32 = [0.000188823; 0.000257837; 0.000339492; 0.000360864; 0.000869312; 0.002464255; 0.009166229; 0.036699104; 0.146715824;;]
times_9965_384_fp64 = [6.2989179e-5; 7.8659849e-5; 0.000171432365; 0.000843469431; 0.003123116; 0.013415864; 0.056932045; 0.230429893; 0.926631256;;]
times_9965_384_fp32 = [6.2599546e-5; 7.6081961e-5; 0.000167520392; 0.000776444309; 0.003347552; 0.014496765; 0.059187747; 0.229520258; 0.893409997;;]

# _, times = run_benchmark_precomputed(benchmark_tlsph, ...)
times_tlsph_h100_fp64 = [0.000221666; 0.000277315; 0.000313635; 0.000581413; 0.001525038; 0.006187035; 0.024247305; 0.097844431; 0.391956402;;]
times_tlsph_h100_fp32 = [0.000211938; 0.000257187; 0.000305539; 0.000313155; 0.000936904; 0.003801156; 0.014689705; 0.058975881; 0.237267362;;]
# The RTX 5090 ran out of memory for the largest problem size with the precomputed NHS.
times_tlsph_5090_fp32 =[0.000142039; 0.000204024; 0.000230026; 0.00025992; 0.001077512; 0.004450762; 0.017486176; 0.070821013;;]
time_tlsph_5090_fp32_42875000 = 0.182653038
# Times for the AMD EPYC 9965 are not yet included here because NUMA-awareness and SIMD
# vectorization are not yet merged into main.

# _, times = run_benchmark_precomputed(benchmark_tlsph_deformation_grad, ...)
times_tlsph_deformation_grad_h100 = [9.0017e-5; 0.000100993; 0.000106274; 0.000113377; 0.000363843; 0.001423087; 0.005557113; 0.022579336; 0.089809181;;]
times_tlsph_deformation_grad_h100_fp32 = [7.7249e-5; 8.6529e-5; 9.2097e-5; 9.3217e-5; 0.000263779; 0.000824937; 0.002983583; 0.012179069; 0.048783029;;]
times_tlsph_deformation_grad_5090_fp32 = [5.6569e-5; 6.894e-5; 7.6672e-5; 8.3602e-5; 0.000313116; 0.00099187; 0.003671176; 0.015459844;;]
time_tlsph_deformation_grad_5090_fp32_42875000 = 0.041021319

function plot_machines_wcsph()
    times = hcat(times_5090_fp32,
                 times_h100_fp32,
                 times_h100_fp64,
                 times_9965_384_fp64,)

    names = ["Nvidia RTX 5090 FP32";;
             "Nvidia H100 FP32";;
             "Nvidia H100 FP64";;
             "2x AMD EPYC 9965 x 192";;]

    plot_benchmark(n_particles_vec, times; label = names,
                   title = "Fluid Interaction Forces (WCSPH)")
end

function plot_machines_tlsph()
    # The RTX 5090 ran out of memory for the largest problem size with the precomputed NHS.
    n_particles_5090 = [1000, 4096, 15625, 64000, 250047, 1030301, 4096000, 16387064, 42875000]
    times_5090 = vcat(times_tlsph_5090_fp32, [time_tlsph_5090_fp32_42875000])

    p = plot_benchmark(n_particles_5090, times_5090; label = "Nvidia RTX 5090 FP32",
                       title = "Structure Interaction Forces (TLSPH)")

    times = hcat(times_tlsph_h100_fp32,
                 times_tlsph_h100_fp64,)

    names = ["Nvidia H100 FP32";;
             "Nvidia H100 FP64";;]

    plot_benchmark!(p, n_particles_vec, times; label = names)
end

function plot_machines_tlsph_deformation_grad()
    # The RTX 5090 ran out of memory for the largest problem size with the precomputed NHS.
    n_particles_5090 = [1000, 4096, 15625, 64000, 250047, 1030301, 4096000, 16387064, 42875000]
    times_5090 = vcat(times_tlsph_deformation_grad_5090_fp32, [time_tlsph_deformation_grad_5090_fp32_42875000])

    p = plot_benchmark(n_particles_5090, times_5090; label = "Nvidia RTX 5090 FP32",
                       title = "Deformation Gradient (TLSPH)")

    times = hcat(times_tlsph_deformation_grad_h100_fp32,
                 times_tlsph_deformation_grad_h100,)

    names = ["Nvidia H100 FP32";;
             "Nvidia H100 FP64";;]

    plot_benchmark!(p, n_particles_vec, times; label = names)
end
