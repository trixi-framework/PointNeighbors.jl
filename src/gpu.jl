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
Adapt.@adapt_structure GridNeighborhoodSearch

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

function Adapt.adapt_structure(to, nhs::PrecomputedNeighborhoodSearch)
    neighbor_lists = Adapt.adapt_structure(to, nhs.neighbor_lists)
    search_radius = Adapt.adapt_structure(to, nhs.search_radius)
    periodic_box = Adapt.adapt_structure(to, nhs.periodic_box)
    neighborhood_search = Adapt.adapt_structure(to, nhs.neighborhood_search)

    return PrecomputedNeighborhoodSearch{ndims(nhs)}(neighbor_lists, search_radius,
                                                     periodic_box, neighborhood_search)
end

function Adapt.adapt_structure(to, cell_list::SpatialHashingCellList{NDIMS}) where {NDIMS}
    (; list_size) = cell_list
    cells = Adapt.adapt_structure(to, cell_list.cells)
    coords = Adapt.adapt_structure(to, cell_list.coords)
    collisions = Adapt.adapt_structure(to, cell_list.collisions)

    return SpatialHashingCellList(NDIMS, cells, coords, collisions, list_size)
end
