module PointNeighbors

using Reexport: @reexport

using Adapt: Adapt
using Atomix: Atomix
using Base: @propagate_inbounds
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
include("nhs_tree.jl")
include("gpu.jl")

export foreach_point_neighbor, foreach_point_neighbor_unsafe,
       foreach_neighbor, foreach_neighbor_unsafe,
       mapreduce_neighbor, mapreduce_neighbor_unsafe
export TrivialNeighborhoodSearch, GridNeighborhoodSearch, PrecomputedNeighborhoodSearch,
       TreeNeighborhoodSearch
export DictionaryCellList, FullGridCellList, SpatialHashingCellList, TreeCellList
export DynamicVectorOfVectors
export ParallelUpdate, SemiParallelUpdate, SerialIncrementalUpdate, SerialUpdate,
       ParallelIncrementalUpdate
export requires_update
export initialize!, update!, initialize_grid!, update_grid!, initialize_tree!, update_tree!
export SerialBackend, PolyesterBackend, ThreadsDynamicBackend, ThreadsStaticBackend,
       default_backend
export PeriodicBox, copy_neighborhood_search

end # module PointNeighbors
