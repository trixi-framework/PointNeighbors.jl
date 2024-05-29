module PointNeighbors

using Reexport: @reexport

using LinearAlgebra: dot
using Polyester: @batch
@reexport using StaticArrays: SVector

include("util.jl")
include("neighborhood_search.jl")
include("nhs_trivial.jl")
include("nhs_grid.jl")
include("cell_lists.jl")

export for_particle_neighbor
export TrivialNeighborhoodSearch, GridNeighborhoodSearch
export initialize!, update!, initialize_grid!, update_grid!

end # module PointNeighbors
