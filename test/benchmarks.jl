# Check that all benchmarks are running without errors.
# Note that these are only smoke tests, not verifying the result.
# Also note that these tests are run without coverage checks, since we want to
# cover everything with unit tests.
@testset verbose=true "Benchmarks" begin
    include("../benchmarks/benchmarks.jl")

    @testset verbose=true "$(length(size))D" for size in [(50,), (10, 10), (5, 5, 5)]
        @testset verbose=true "`benchmark_count_neighbors`" begin
            @trixi_test_nowarn run_benchmark_default(benchmark_count_neighbors, size, 2)
            @trixi_test_nowarn run_benchmark_gpu(benchmark_count_neighbors, size, 2)
        end

        @testset verbose=true "`benchmark_n_body`" begin
            @trixi_test_nowarn run_benchmark_default(benchmark_n_body, size, 2)
            @trixi_test_nowarn run_benchmark_gpu(benchmark_n_body, size, 2)
        end

        @testset verbose=true "`benchmark_wcsph`" begin
            @trixi_test_nowarn run_benchmark_default(benchmark_wcsph, size, 2)
            @trixi_test_nowarn run_benchmark_gpu(benchmark_wcsph, size, 2)
        end

        @testset verbose=true "`benchmark_wcsph_fp32`" begin
            @trixi_test_nowarn run_benchmark_default(benchmark_wcsph_fp32, size, 2)
            @trixi_test_nowarn run_benchmark_gpu(benchmark_wcsph_fp32, size, 2)
        end

        @testset verbose=true "`benchmark_tlsph`" begin
            @trixi_test_nowarn run_benchmark_default(benchmark_tlsph, size, 2)
            @trixi_test_nowarn run_benchmark_gpu(benchmark_tlsph, size, 2)
        end

        @testset verbose=true "`benchmark_initialize`" begin
            @trixi_test_nowarn run_benchmark_default(benchmark_initialize, size, 2)
            @trixi_test_nowarn run_benchmark_gpu(benchmark_initialize, size, 2)
        end

        @testset verbose=true "`benchmark_update_alternating`" begin
            @trixi_test_nowarn run_benchmark_default(benchmark_update_alternating, size, 2)
            @trixi_test_nowarn run_benchmark_gpu(benchmark_update_alternating, size, 2)
        end
    end
end;
