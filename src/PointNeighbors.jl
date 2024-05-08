module PointNeighbors

using Reexport: @reexport

using LinearAlgebra: dot
using Polyester: @batch
@reexport using StaticArrays: SVector

include("util.jl")
include("neighborhood_search.jl")
include("trivial_nhs.jl")
include("grid_nhs.jl")

export for_particle_neighbor
export TrivialNeighborhoodSearch, GridNeighborhoodSearch
export initialize!, update!

end # module PointNeighbors
