@testset verbose=true "`DynamicVectorOfVectors`" begin
    # Test different types by defining a function to convert to this type
    types = [Int32, Float64, i -> (i, i)]

    @testset verbose=true "Eltype $(eltype(type(1)))" for type in types
        ELTYPE = typeof(type(1))
        vov_ref = Vector{Vector{ELTYPE}}()
        vov = PointNeighbors.DynamicVectorOfVectors{ELTYPE}(max_outer_length = 20,
                                                            max_inner_length = 4)

        function verify(vov, vov_ref)
            @test length(vov) == length(vov_ref)
            @test eachindex(vov) == eachindex(vov_ref)
            @test axes(vov) == axes(vov_ref)

            @test_throws BoundsError vov[0]
            @test_throws BoundsError vov[length(vov) + 1]

            for i in eachindex(vov_ref)
                @test vov[i] == vov_ref[i]
            end
        end

        @testset "Initial state" begin
            # Test internal size
            @test size(vov.backend) == (4, 20)

            # Initial check
            verify(vov, vov_ref)
        end

        @testset "First push!" begin
            push!(vov_ref, type.([1, 3, 2]))
            push!(vov, type.([1, 3, 2]))

            verify(vov, vov_ref)
        end

        @testset "push! multiple items" begin
            push!(vov_ref, type.([4]), type.([5, 6, 7, 8]))
            push!(vov, type.([4]), type.([5, 6, 7, 8]))

            verify(vov, vov_ref)
        end

        @testset "push! to an inner vector" begin
            push!(vov_ref[1], type(12))
            PointNeighbors.pushat!(vov, 1, type(12))

            verify(vov, vov_ref)
        end

        @testset "push! overflow" begin
            error_ = ErrorException("cell list is full. Use a larger `max_points_per_cell`.")
            @test_throws error_ PointNeighbors.pushat!(vov, 1, type(13))

            verify(vov, vov_ref)
        end

        @testset "deleteat!" begin
            # Delete entry of inner vector. Note that this changes the order of the elements.
            deleteat!(vov_ref[3], 2)
            PointNeighbors.deleteatat!(vov, 3, 2)

            @test vov_ref[3] == type.([5, 7, 8])
            @test vov[3] == type.([5, 8, 7])

            # Delete second to last entry
            deleteat!(vov_ref[3], 2)
            PointNeighbors.deleteatat!(vov, 3, 2)

            @test vov_ref[3] == type.([5, 8])
            @test vov[3] == type.([5, 7])

            # Delete last entry
            deleteat!(vov_ref[3], 2)
            PointNeighbors.deleteatat!(vov, 3, 2)

            # Now they are identical again
            verify(vov, vov_ref)

            # Delete the remaining entry of this vector
            deleteat!(vov_ref[3], 1)
            PointNeighbors.deleteatat!(vov, 3, 1)

            verify(vov, vov_ref)
        end

        # Skip for Tuples
        if ELTYPE <: Number
            @testset "sorteach!" begin
                # Make sure that the first inner vector is unsorted.
                # If this fails, make sure the tests above don't yield a sorted vector.
                @test vov[1] != sort(vov[1])

                PointNeighbors.sorteach!(vov)
                PointNeighbors.sorteach!(vov_ref)

                @test vov[1] == sort(vov[1])

                verify(vov, vov_ref)
            end
        end

        @testset "empty!" begin
            empty!(vov_ref)
            empty!(vov)

            verify(vov, vov_ref)
        end
    end
end

@testset verbose=true "`CompactVectorOfVectors`" begin
    n_bins = 10
    n_values = 200
    bins = rand(Random.Xoshiro(1), 1:n_bins, n_values)
    value_to_bin(value) = bins[value]

    # Reference: the values of each bin, sorted in increasing order
    function vov_reference(active_values)
        vov_ref = [Int32[] for _ in 1:n_bins]
        for value in sort(collect(active_values))
            push!(vov_ref[bins[value]], value)
        end

        return vov_ref
    end

    backends = [SerialBackend(), PolyesterBackend(), ThreadsDynamicBackend()]
    active_values_ = [
        Base.OneTo(n_values),
        11:150,
        Random.randperm(Random.Xoshiro(2), n_values)[1:120]
    ]
    names = ["all values", "range of values", "random subset of values"]

    @testset verbose=true "$(names[i]) with $(nameof(typeof(backend)))" for i in
                                                                            eachindex(names),
                                                                            backend in
                                                                            backends

        active_values = active_values_[i]
        vov = PointNeighbors.CompactVectorOfVectors{Int32}(; n_bins)

        @test length(vov) == n_bins
        @test all(isempty, vov)

        # Update twice to make sure that the update doesn't depend on the previous state
        PointNeighbors.update!(vov, value -> 1, n_values, backend)
        PointNeighbors.update!(vov, value_to_bin, n_values, backend; active_values)

        vov_ref = vov_reference(active_values)
        @test length(vov) == n_bins
        @test_throws BoundsError vov[0]
        @test_throws BoundsError vov[n_bins + 1]
        for bin in 1:n_bins
            @test vov[bin] == vov_ref[bin]
        end

        # `values` is a permutation of all values, sorted by bin,
        # followed by the inactive values in increasing order
        inactive_values = setdiff(1:n_values, active_values)
        @test vov.values == vcat(vov_ref..., inactive_values)
    end
end
