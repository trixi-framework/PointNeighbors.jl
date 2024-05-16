using Random
using StaticArrays

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

    perturb!(coordinates, perturbation_factor_position * 0.5)

    return coordinates
end

function perturb!(data, amplitude)
    for i in eachindex(data)
        # Perturbation in the interval (-amplitude, amplitude)
        data[i] += 2 * amplitude * rand() - amplitude
    end

    return data
end
