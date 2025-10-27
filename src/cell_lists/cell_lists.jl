abstract type AbstractCellList end

# For the `DictionaryCellList`, this is a `KeySet`, which has to be `collect`ed first to be
# able to be used in a threaded loop.
@inline each_cell_index_threadable(cell_list::AbstractCellList) = each_cell_index(cell_list)

include("dictionary.jl")
include("full_grid.jl")
include("spatial_hashing.jl")
