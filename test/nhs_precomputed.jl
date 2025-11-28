@testset verbose=true "PrecomputedNeighborhoodSearch" begin
    # Test regular vs transposed backend
    nhs = PrecomputedNeighborhoodSearch{2}(transpose_backend = false, n_points = 2)

    # Add test neighbors
    neighbor_lists = nhs.neighbor_lists
    @test PointNeighbors.transposed_backend(neighbor_lists) == false
    neighbor_lists.backend .= 0
    PointNeighbors.pushat!(neighbor_lists, 1, 101)
    PointNeighbors.pushat!(neighbor_lists, 1, 102)
    PointNeighbors.pushat!(neighbor_lists, 2, 201)
    PointNeighbors.pushat!(neighbor_lists, 2, 202)
    PointNeighbors.pushat!(neighbor_lists, 2, 203)

    # Check that neighbors are next to each other in memory
    pointer_ = pointer(neighbor_lists.backend)
    @test unsafe_load(pointer_, 1) == 101
    @test unsafe_load(pointer_, 2) == 102

    # Transposed backend
    nhs = PrecomputedNeighborhoodSearch{2}(transpose_backend = true, n_points = 2)

    # Add test neighbors
    neighbor_lists = nhs.neighbor_lists
    @test PointNeighbors.transposed_backend(neighbor_lists) == true
    neighbor_lists.backend .= 0
    PointNeighbors.pushat!(neighbor_lists, 1, 101)
    PointNeighbors.pushat!(neighbor_lists, 1, 102)
    PointNeighbors.pushat!(neighbor_lists, 2, 201)
    PointNeighbors.pushat!(neighbor_lists, 2, 202)
    PointNeighbors.pushat!(neighbor_lists, 2, 203)

    # Check that first neighbors are next to each other in memory
    @test neighbor_lists.backend isa PermutedDimsArray
    pointer_ = pointer(neighbor_lists.backend.parent)
    @test unsafe_load(pointer_, 1) == 101
    @test unsafe_load(pointer_, 2) == 201
end
