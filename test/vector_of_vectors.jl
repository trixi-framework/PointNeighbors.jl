@testset verbose=true "`DynamicVectorOfVectors`" begin
    # Test different types by defining a function to convert to this type
    types = [Int32, Float64, i -> (i, i)]

    @testset verbose=true "Eltype $(eltype(type(1)))" for type in types
        ELTYPE = typeof(type(1))
        vov_ref = Vector{Vector{ELTYPE}}()
        vov = PointNeighbors.DynamicVectorOfVectors{ELTYPE}(max_outer_length = 20,
                                                            max_inner_length = 4)

        # Test internal size
        @test size(vov.backend) == (4, 20)

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

        # Initial check
        verify(vov, vov_ref)

        # First `push!`
        push!(vov_ref, type.([1, 2, 3]))
        push!(vov, type.([1, 2, 3]))

        verify(vov, vov_ref)

        # `push!` multiple items
        push!(vov_ref, type.([4]), type.([5, 6, 7, 8]))
        push!(vov, type.([4]), type.([5, 6, 7, 8]))

        verify(vov, vov_ref)

        # `push!` to an inner vector
        push!(vov_ref[1], type(12))
        PointNeighbors.pushat!(vov, 1, type(12))

        # `push!` overflow
        error_ = ErrorException("cell list is full. Use a larger `max_points_per_cell`.")
        @test_throws error_ PointNeighbors.pushat!(vov, 1, type(13))

        verify(vov, vov_ref)

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

        # `empty!`
        empty!(vov_ref)
        empty!(vov)

        verify(vov, vov_ref)
    end
end
