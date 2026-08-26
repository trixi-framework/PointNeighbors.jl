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
include("gpu.jl")

export foreach_point_neighbor, foreach_point_neighbor_unsafe,
       foreach_neighbor, foreach_neighbor_unsafe,
       mapreduce_neighbor, mapreduce_neighbor_unsafe
export TrivialNeighborhoodSearch, GridNeighborhoodSearch, PrecomputedNeighborhoodSearch
export DictionaryCellList, FullGridCellList, SpatialHashingCellList
export DynamicVectorOfVectors
export ParallelUpdate, SemiParallelUpdate, SerialIncrementalUpdate, SerialUpdate,
       ParallelIncrementalUpdate
export requires_update
export initialize!, update!, initialize_grid!, update_grid!
export SerialBackend, PolyesterBackend, ThreadsDynamicBackend, ThreadsStaticBackend,
       default_backend
export PeriodicBox, copy_neighborhood_search
export CellListMapNeighborhoodSearch

"""
    CellListMapNeighborhoodSearch(NDIMS; search_radius = 1.0, points_equal_neighbors = false)

Neighborhood search based on the package
[CellListMap.jl](https://github.com/m3g/CellListMap.jl).
This is only available when CellListMap.jl is loaded (this function is implemented in the
package extension `PointNeighborsCellListMapExt`).

!!! warning "Parallelization is not point-partitioned"
    Every other `AbstractNeighborhoodSearch` parallelizes [`foreach_point_neighbor`](@ref)
    by splitting the outer loop over `points`, guaranteeing that a given point index is only
    ever touched by one thread. This is what makes unsynchronized per-point accumulation in
    the callback (e.g. `dv[:, i] += ...`, `n_neighbors[i] += 1`), the idiomatic pattern used
    throughout PointNeighbors.jl and TrixiParticles.jl, safe under any parallel backend.

    `CellListMapNeighborhoodSearch` does not follow this: it delegates the parallel
    traversal to CellListMap.jl, which parallelizes by *cell-pair batches*, not by point.
    A single point's neighbors can be split across batches handled by different threads, so
    the callback for the same point index can run concurrently on more than one thread.
    Unsynchronized per-point accumulation is therefore **not safe** with this implementation
    under any `parallelization_backend` other than `SerialBackend()` — it can silently lose
    updates (or, for callbacks that resize a shared container, throw a
    `ConcurrencyViolationError`). Either use `SerialBackend()`, or make sure the callback
    itself is safe against concurrent calls for the same point index (e.g. using atomics).
"""
function CellListMapNeighborhoodSearch end

end # module PointNeighbors
