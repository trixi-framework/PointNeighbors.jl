module PointNeighbors

using Reexport: @reexport

using Atomix: Atomix
using LinearAlgebra: dot
using Polyester: @batch
@reexport using StaticArrays: SVector

include("util.jl")
include("vector_of_vectors.jl")
include("neighborhood_search.jl")
include("nhs_trivial.jl")
include("cell_lists/cell_lists.jl")
include("nhs_grid.jl")
include("nhs_precomputed.jl")

export foreach_point_neighbor, foreach_neighbor
export TrivialNeighborhoodSearch, GridNeighborhoodSearch, PrecomputedNeighborhoodSearch
export DictionaryCellList, FullGridCellList
export initialize!, update!, initialize_grid!, update_grid!
export PeriodicBox, copy_neighborhood_search

end # module PointNeighbors
