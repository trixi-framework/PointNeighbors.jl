using Random

# Generate a rectangular point cloud, optionally with a perturbation in the point positions
function point_cloud(n_points_per_dimension;
                     seed = 1, perturbation_factor_position = 1.0)
    # Fixed seed to ensure reproducibility
    Random.seed!(seed)

    n_dims = length(n_points_per_dimension)
    coordinates = Array{Float64}(undef, n_dims, prod(n_points_per_dimension))
    cartesian_indices = CartesianIndices(n_points_per_dimension)

    for i in axes(coordinates, 2)
        coordinates[:, i] .= Tuple(cartesian_indices[i])
    end

    # A standard deviation of 0.05 in the particle coordinates
    # corresponds to a standard deviation of 2 in the number of neighbors for a 300 x 100
    # grid, 1.6 for a 600 x 200 grid and 1.26 for a 1200 x 400 grid.
    # This is consistent with the standard deviation in a vortex street simulation.
    # The benchmark results are also consistent with the timer output of the simulation.
    perturb!(coordinates, perturbation_factor_position * 0.05)

    return coordinates
end

function perturb!(data, std_deviation)
    for i in eachindex(data)
        data[i] += std_deviation * randn()
    end

    return data
end
