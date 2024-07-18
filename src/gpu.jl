# Adapt.jl provides a function `adapt(to, x)`, which adapts a value `x` to `to`.
# In practice, this means that we can use `adapt(CuArray, system)` to adapt a system to
# the `CuArray` type.
# What this does is that it converts all `Array`s inside this system to `CuArray`s,
# therefore copying them to the GPU.
# In order to run a simulation on a GPU, we want to call `adapt(T, nhs)` to adapt the
# neighborhood search `nhs` to the GPU array type `T` (e.g. `CuArray`).
#
# `Adapt.@adapt_structure` automatically generates the `adapt` function for our custom types.
Adapt.@adapt_structure FullGridCellList
Adapt.@adapt_structure DynamicVectorOfVectors

# `adapt(CuArray, ::SVector)::SVector`, but `adapt(Array, ::SVector)::Vector`.
# We don't want to change the type of the `SVector` here.
function Adapt.adapt_structure(to::typeof(Array), svector::SVector)
    return svector
end

# `adapt(CuArray, ::UnitRange)::UnitRange`, but `adapt(Array, ::UnitRange)::Vector`.
# We don't want to change the type of the `UnitRange` here.
function Adapt.adapt_structure(to::typeof(Array), range::UnitRange)
    return range
end

function Adapt.adapt_structure(to, nhs::GridNeighborhoodSearch)
    (; search_radius, periodic_box, n_cells, cell_size) = nhs

    cell_list = Adapt.adapt_structure(to, nhs.cell_list)
    update_buffer = Adapt.adapt_structure(to, nhs.update_buffer)

    return GridNeighborhoodSearch(cell_list, search_radius, periodic_box, n_cells,
                                  cell_size, update_buffer, nhs.update_strategy)
end

# This is useful to pass the backend directly to `@threaded`
KernelAbstractions.get_backend(backend::KernelAbstractions.Backend) = backend
