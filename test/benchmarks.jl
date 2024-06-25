# Check that all benchmarks are running without errors.
# Note that these are only smoke tests, not verifying the result.
# Also note that these tests are run without coverage checks, since we want to
# cover everything with unit tests.
@testset verbose=true "Benchmarks" begin
    include("../benchmarks/benchmarks.jl")

    @testset verbose=true "$(length(size))D" for size in [(50,), (10, 10), (5, 5, 5)]
        @testset verbose=true "`benchmark_count_neighbors`" begin
            @test_nowarn_mod plot_benchmarks(benchmark_count_neighbors, size, 2)
        end

        @testset verbose=true "`benchmark_n_body`" begin
            @test_nowarn_mod plot_benchmarks(benchmark_n_body, size, 2)
        end
    end
end;
