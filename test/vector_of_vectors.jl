@testset verbose=true "`StaticVectorOfVectors`" begin
    n_bins = 3
    values = [2, 3, 5, 1, 4]
    vov = PointNeighbors.CompactVectorOfVectors{Int}(n_bins = n_bins)
    resize!(vov.values, length(values))
    vov.values .= eachindex(values)

    # Fill values and assign bins
    f(x) = x % n_bins + 1
    PointNeighbors.update!(vov, f)

    # Test bin sizes
    for i in 1:n_bins
        bin = vov[i]
        @test all(f(x) == i for x in bin)
    end

    # Test that all values are present
    @test sort(vcat([collect(vov[i]) for i in 1:n_bins]...)) == sort(vov.values)
end
