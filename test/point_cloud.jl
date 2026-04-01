using Random

# Generate a rectangular point cloud, optionally with a perturbation in the point positions
function point_cloud(n_points_per_dimension;
                     seed = 1, perturbation_factor_position = 1.0, shuffle = false)
    # Fixed seed to ensure reproducibility
    Random.seed!(seed)

    n_dims = length(n_points_per_dimension)
    coordinates = Array{Float64}(undef, n_dims, prod(n_points_per_dimension))
    cartesian_indices = CartesianIndices(n_points_per_dimension)

    # Extra data structures for the sorting code below
    cell_coords = Vector{SVector{n_dims, Int}}(undef, size(coordinates, 2))
    cell_size = ntuple(dim -> 4.0, n_dims)

    for i in axes(coordinates, 2)
        point_coords = SVector(Tuple(cartesian_indices[i]))
        coordinates[:, i] .= point_coords
        cell_coords[i] = PointNeighbors.cell_coords(point_coords, nothing, nothing,
                                                    cell_size) .+ 1
    end

    # A standard deviation of 0.05 in the particle coordinates
    # corresponds to a standard deviation of 2 in the number of neighbors for a 300 x 100
    # grid, 1.6 for a 600 x 200 grid and 1.26 for a 1200 x 400 grid.
    # This is consistent with the standard deviation in a vortex street simulation.
    # The benchmark results are also consistent with the timer output of the simulation.
    perturb!(coordinates, perturbation_factor_position * 0.05)

    # Sort by Z index (with `using Morton`)
    # permutation = sortperm(cell_coords, by = c -> cartesian2morton(c))

    # Sort by linear cell index
    # permutation = sortperm(cell_coords)
    # coordinates .= coordinates[:, permutation]

    if shuffle
        # Sort randomly
        permutation = shuffle(axes(coordinates, 2))

        coordinates .= coordinates[:, permutation]
    end

    return coordinates
end

function perturb!(data, std_deviation)
    for i in eachindex(data)
        data[i] += std_deviation * randn()
    end

    return data
end
