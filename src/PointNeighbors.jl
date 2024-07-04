module PointNeighbors

using Reexport: @reexport

using Adapt: Adapt
using Atomix: Atomix
using GPUArraysCore: AbstractGPUArray
using KernelAbstractions: KernelAbstractions, @kernel, @index
using LinearAlgebra: dot
using Polyester: Polyester
@reexport using StaticArrays: SVector

include("util.jl")
include("vector_of_vectors.jl")
include("neighborhood_search.jl")
include("nhs_trivial.jl")
include("cell_lists/cell_lists.jl")
include("nhs_grid.jl")
include("nhs_precomputed.jl")
include("gpu.jl")

export foreach_point_neighbor, foreach_neighbor
export TrivialNeighborhoodSearch, GridNeighborhoodSearch, PrecomputedNeighborhoodSearch,
       CellListMapNeighborhoodSearch
export DictionaryCellList, FullGridCellList
export ParallelUpdate, SemiParallelUpdate, SerialUpdate
export initialize!, update!, initialize_grid!, update_grid!
export PeriodicBox, copy_neighborhood_search

end # module PointNeighbors
