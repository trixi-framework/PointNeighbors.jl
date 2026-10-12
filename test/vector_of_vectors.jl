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
    # Three vectors with space for 2, 0 and 3 values
    vov = CompactVectorOfVectors{Int32}(max_outer_length = 3, max_total_length = 5)
    PointNeighbors.set_offset!(vov, 1, 0)
    PointNeighbors.set_offset!(vov, 2, 2)
    PointNeighbors.set_offset!(vov, 3, 2)

    @test length(vov) == 3
    @test all(isempty, vov)

    PointNeighbors.pushat!(vov, 3, 31)
    PointNeighbors.pushat!(vov, 1, 11)
    PointNeighbors.pushat_atomic!(vov, 3, 32)
    PointNeighbors.pushat!(vov, 1, 12)
    PointNeighbors.pushat_atomic!(vov, 3, 33)

    @test vov[1] == [11, 12]
    @test isempty(vov[2])
    @test vov[3] == [31, 32, 33]

    # All values are stored contiguously without gaps
    @test vov.values == [11, 12, 31, 32, 33]

    # Empty the third vector and fill it again
    PointNeighbors.emptyat!(vov, 3)
    @test isempty(vov[3])
    PointNeighbors.pushat!(vov, 3, 34)
    @test vov[3] == [34]
    @test vov[1] == [11, 12]
end
