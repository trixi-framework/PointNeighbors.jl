using Plots

"""
    plot_benchmark(n_particles_vec, times; kwargs...)

Plot the results of a benchmark run with [`run_benchmark`](@ref).
Note that the arguments are the outputs of that function.

# Arguments
- `n_particles_vec`: Vector containing the number of particles for each iteration.
- `times`:           Matrix containing the runtimes for each neighborhood search and iteration.

# Keywords
Keyword arguments are passed to `Plots.plot`. For example, use `title = "My title"`.

# Examples
```julia
include(joinpath(pkgdir(PointNeighbors),  "benchmarks", "benchmarks.jl"));
include(joinpath(pkgdir(PointNeighbors),  "benchmarks", "plot_benchmarks.jl"));

n_particles_vec, times = run_benchmark_default(benchmark_count_neighbors, (10, 10), 3)
plot_benchmark(n_particles_vec, times; title = "Count neighbors benchmark")
```
"""
function plot_benchmark(n_particles_vec, times; kwargs...)
    p = plot()
    plot_benchmark!(p, n_particles_vec, times; kwargs...)
end

function plot_benchmark!(p, n_particles_vec, times; kwargs...)
    function format_n_particles(n)
        if n >= 1_000_000
            return "$(round(Int, n / 1_000_000))M"
        elseif n >= 1_000
            return "$(round(Int, n / 1_000))k"
        else
            return string(n)
        end
    end
    xticks = format_n_particles.(n_particles_vec)

    plot!(p, n_particles_vec, n_particles_vec ./ times .* 1e-6;
          xaxis = :log, xticks = (n_particles_vec, xticks), linewidth = 2,
          # Make sure the plot starts at y = 0.
          ylimits = (0, Inf), widen = true,
          xlabel = "#particles", ylabel = "million particles processed per second",
          legend = :outerright, size = (700, 350), dpi = 600, margin = 4 * Plots.mm,
          palette = palette(:tab10), kwargs...)
end

# Plot benchmarks for different machines.
# Run these benchmarks like this. The type of `search_radius_factor` determines
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
benchmark_runtimes = (n_particles = [
                          1000, 4096, 15625, 64000, 250047, 1030301, 4096000, 16387064,
                          65450827],
                      # _, times = run_benchmark_full_grid(benchmark_wcsph, ...)
                      wcsph_h100_fp64 = [0.000337699; 0.000401732; 0.000480261; 0.000522277; 0.001476047; 0.006047292; 0.023739696; 0.099076906; 0.397741399;;],
                      wcsph_h100_fp32 = [0.00025341; 0.000311011; 0.000368099; 0.000390308; 0.001002314; 0.003407714; 0.013074915; 0.052667697; 0.212927902;;],
                      wcsph_5090_fp32 = [0.000188823; 0.000257837; 0.000339492; 0.000360864; 0.000869312; 0.002464255; 0.009166229; 0.036699104; 0.146715824;;],
                      wcsph_9965_384_fp64 = [6.2989179e-5; 7.8659849e-5; 0.000171432365; 0.000843469431; 0.003123116; 0.013415864; 0.056932045; 0.230429893; 0.926631256;;],
                      wcsph_9965_384_fp32 = [6.2599546e-5; 7.6081961e-5; 0.000167520392; 0.000776444309; 0.003347552; 0.014496765; 0.059187747; 0.229520258; 0.893409997;;],
                      # _, times = run_benchmark_precomputed(benchmark_tlsph, ...)
                      # Times for the AMD EPYC 9965 are not yet included here because NUMA-awareness and SIMD
                      # vectorization are not yet merged into main.
                      tlsph_h100_fp64 = [0.000221666; 0.000277315; 0.000313635; 0.000581413; 0.001525038; 0.006187035; 0.024247305; 0.097844431; 0.391956402;;],
                      tlsph_h100_fp32 = [0.000211938; 0.000257187; 0.000305539; 0.000313155; 0.000936904; 0.003801156; 0.014689705; 0.058975881; 0.237267362;;],
                      # The RTX 5090 ran out of memory for the largest problem size with the precomputed NHS.
                      n_particles_tlsph_5090 = [
                          1000, 4096, 15625, 64000, 250047, 1030301, 4096000, 16387064,
                          42875000],
                      tlsph_5090_fp32 = [0.000142039; 0.000204024; 0.000230026; 0.00025992; 0.001077512; 0.004450762; 0.017486176; 0.070821013; 0.182653038;;],
                      # _, times = run_benchmark_precomputed(benchmark_tlsph_deformation_grad, ...)
                      tlsph_deformation_grad_h100_fp64 = [9.0017e-5; 0.000100993; 0.000106274; 0.000113377; 0.000363843; 0.001423087; 0.005557113; 0.022579336; 0.089809181;;],
                      tlsph_deformation_grad_h100_fp32 = [7.7249e-5; 8.6529e-5; 9.2097e-5; 9.3217e-5; 0.000263779; 0.000824937; 0.002983583; 0.012179069; 0.048783029;;],
                      # The RTX 5090 ran out of memory for the largest problem size with the precomputed NHS.
                      n_particles_tlsph_deformation_grad_5090 = [
                          1000, 4096, 15625, 64000, 250047, 1030301, 4096000, 16387064,
                          42875000],
                      tlsph_deformation_grad_5090_fp32 = [5.6569e-5; 6.894e-5; 7.6672e-5; 8.3602e-5; 0.000313116; 0.00099187; 0.003671176; 0.015459844; 0.041021319;;],
                      # Benchmarking the difference between implementations on an
                      # Intel Xeon W9-3475X (x36).
                      # _, times = run_benchmark_default(benchmark_wcsph, ...)
                      wcsph_w9_3475x_dictionary = [9.539e-5; 0.000507708; 0.001999838; 0.008442291; 0.034369512; 0.145935124; 0.581081578; 2.341472385; 9.445755411;;],
                      wcsph_w9_3475x_fullgrid = [6.0765e-5; 0.000402875; 0.001569317; 0.006578832; 0.026040483; 0.108027132; 0.434874618; 1.755279267; 7.07428452;;],
                      wcsph_w9_3475x_precomputed = [3.7762e-5; 0.000148563; 0.00054964; 0.002274807; 0.00899912; 0.037950566; 0.151298558; 0.607103175; 2.440462785;;],
                      # NaN values are just placeholders because the benchmark took too long.
                      wcsph_w9_3475x_trivial = [6.3518e-5; 0.000669186; 0.008641863; 0.135930064; 2.468379663; 43.607518925; NaN; NaN; NaN;;],
                      # Benchmarking the difference between update strategies on an
                      # Intel Xeon W9-3475X (x36).
                      # _, times = run_benchmark_updates((10, 10, 10), 9)
                      update_w9_3475x_parallel = [1.42395e-5; 1.9205e-5; 3.3569e-5; 7.6216e-5; 0.0001972665; 0.000657154; 0.002366056; 0.008558718; 0.0328478125;;],
                      update_w9_3475x_parallel_incremental = [1.05465e-5; 1.41955e-5; 2.36415e-5; 6.3423e-5; 0.000185218; 0.0007833225; 0.0052182165; 0.025595217; 0.103340834;;],
                      update_w9_3475x_semi_parallel = [1.00805e-5; 1.78505e-5; 4.28215e-5; 0.000153458; 0.0006931965; 0.0026806445; 0.0158252845; 0.079710013; 0.329653309;;],
                      update_w9_3475x_precomputed = [0.000156829; 0.0005617385; 0.0021646195; 0.009024255; 0.0354495175; 0.1443668715; 0.5821228515; 2.341333004; 9.381562997;;])

function plot_machines_wcsph()
    times = hcat(benchmark_runtimes.wcsph_5090_fp32,
                 benchmark_runtimes.wcsph_h100_fp32,
                 benchmark_runtimes.wcsph_h100_fp64,
                 benchmark_runtimes.wcsph_9965_384_fp64)

    names = ["Nvidia RTX 5090 FP32";;
             "Nvidia H100 FP32";;
             "Nvidia H100 FP64";;
             "2x AMD EPYC 9965 x 192";;]

    plot_benchmark(benchmark_runtimes.n_particles, times; label = names,
                   title = "Fluid Interaction Forces (WCSPH)")
end

function plot_machines_tlsph()
    p = plot_benchmark(benchmark_runtimes.n_particles_tlsph_5090,
                       benchmark_runtimes.tlsph_5090_fp32;
                       label = "Nvidia RTX 5090 FP32",
                       title = "Structure Interaction Forces (TLSPH)")

    times = hcat(benchmark_runtimes.tlsph_h100_fp32,
                 benchmark_runtimes.tlsph_h100_fp64)

    names = ["Nvidia H100 FP32";;
             "Nvidia H100 FP64";;]

    plot_benchmark!(p, benchmark_runtimes.n_particles, times; label = names)
end

function plot_machines_tlsph_deformation_grad()
    p = plot_benchmark(benchmark_runtimes.n_particles_tlsph_deformation_grad_5090,
                       benchmark_runtimes.tlsph_deformation_grad_5090_fp32;
                       label = "Nvidia RTX 5090 FP32",
                       title = "Deformation Gradient (TLSPH)")

    times = hcat(benchmark_runtimes.tlsph_deformation_grad_h100_fp32,
                 benchmark_runtimes.tlsph_deformation_grad_h100_fp64)

    names = ["Nvidia H100 FP32";;
             "Nvidia H100 FP64";;]

    plot_benchmark!(p, benchmark_runtimes.n_particles, times; label = names)
end

function plot_implementations_wcsph()
    times = hcat(benchmark_runtimes.wcsph_w9_3475x_precomputed,
                 benchmark_runtimes.wcsph_w9_3475x_fullgrid,
                 benchmark_runtimes.wcsph_w9_3475x_dictionary,
                 benchmark_runtimes.wcsph_w9_3475x_trivial)

    names = ["PrecomputedNeighborhoodSearch";;
             "GNHS & FullGridCellList";;
             "GridNeighborhoodSearch";;
             "TrivialNeighborhoodSearch";;]

    plot_benchmark(benchmark_runtimes.n_particles, times; label = names,
                   title = "WCSPH on Intel Xeon W9-3475X (x36)")
end

function plot_update_strategies()
    times = hcat(benchmark_runtimes.update_w9_3475x_parallel,
                 benchmark_runtimes.update_w9_3475x_parallel_incremental,
                 benchmark_runtimes.update_w9_3475x_semi_parallel,
                 benchmark_runtimes.update_w9_3475x_precomputed)

    names = ["GNHS & ParallelUpdate";;
             "GNHS & ParallelIncrementalUpdate";;
             "GNHS & SemiParallelUpdate";;
             "PrecomputedNeighborhoodSearch";;]

    plot_benchmark(benchmark_runtimes.n_particles, times; label = names,
                   title = "Update strategies on Intel Xeon W9-3475X (x36)")
end
