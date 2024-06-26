include("test_util.jl")

const POINTNEIGHBORS_TEST = lowercase(get(ENV, "POINTNEIGHBORS_TEST", "all"))

@testset verbose=true "PointNeighbors.jl Tests" begin
    if POINTNEIGHBORS_TEST in ("all", "unit")
        include("unittest.jl")
    end

    if POINTNEIGHBORS_TEST in ("all", "benchmarks")
        include("benchmarks.jl")
    end
end;
