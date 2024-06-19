var documenterSearchIndex = {"docs":
[{"location":"license/","page":"License","title":"License","text":"EditURL = \"https://github.com/trixi-framework/PointNeighbors.jl/blob/main/LICENSE.md\"","category":"page"},{"location":"license/#License","page":"License","title":"License","text":"","category":"section"},{"location":"license/","page":"License","title":"License","text":"MIT LicenseCopyright (c) 2023-present The TrixiParticles.jl Authors (see Authors) \nCopyright (c) 2023-present Helmholtz-Zentrum hereon GmbH, Institute of Surface Science \n \nPermission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the \"Software\"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.","category":"page"},{"location":"authors/","page":"Authors","title":"Authors","text":"EditURL = \"https://github.com/trixi-framework/PointNeighbors.jl/blob/main/AUTHORS.md\"","category":"page"},{"location":"authors/#Authors","page":"Authors","title":"Authors","text":"","category":"section"},{"location":"authors/","page":"Authors","title":"Authors","text":"This package is maintained by the authors of TrixiParticles.jl. For a full list of authors, see AUTHORS.md in the TrixiParticles.jl repository. These authors form \"The TrixiParticles.jl Authors\", as mentioned under License.","category":"page"},{"location":"reference/#API-reference","page":"API reference","title":"API reference","text":"","category":"section"},{"location":"reference/","page":"API reference","title":"API reference","text":"CurrentModule = PointNeighbors","category":"page"},{"location":"reference/","page":"API reference","title":"API reference","text":"Modules = [PointNeighbors]","category":"page"},{"location":"reference/#PointNeighbors.GridNeighborhoodSearch","page":"API reference","title":"PointNeighbors.GridNeighborhoodSearch","text":"GridNeighborhoodSearch{NDIMS}(search_radius, n_particles; periodic_box_min_corner=nothing,\n                              periodic_box_max_corner=nothing, threaded_nhs_update=true)\n\nSimple grid-based neighborhood search with uniform search radius. The domain is divided into a regular grid. For each (non-empty) grid cell, a list of particles in this cell is stored. Instead of representing a finite domain by an array of cells, a potentially infinite domain is represented by storing cell lists in a hash table (using Julia's Dict data structure), indexed by the cell index tuple\n\nleft( leftlfloor fracxd rightrfloor leftlfloor fracyd rightrfloor right) quad textor quad\nleft( leftlfloor fracxd rightrfloor leftlfloor fracyd rightrfloor leftlfloor fraczd rightrfloor right)\n\nwhere x y z are the space coordinates and d is the search radius.\n\nTo find particles within the search radius around a point, only particles in the neighboring cells are considered.\n\nSee also (Chalela et al., 2021), (Ihmsen et al. 2011, Section 4.4).\n\nAs opposed to (Ihmsen et al. 2011), we do not sort the particles in any way, since not sorting makes our implementation a lot faster (although less parallelizable).\n\nArguments\n\nNDIMS:          Number of dimensions.\nsearch_radius:  The uniform search radius.\nn_particles:    Total number of particles.\n\nKeywords\n\nperiodic_box_min_corner:    In order to use a (rectangular) periodic domain, pass the                               coordinates of the domain corner in negative coordinate                               directions.\nperiodic_box_max_corner:    In order to use a (rectangular) periodic domain, pass the                               coordinates of the domain corner in positive coordinate                               directions.\nthreaded_nhs_update=true:              Can be used to deactivate thread parallelization in the neighborhood search update.                               This can be one of the largest sources of variations between simulations                               with different thread numbers due to particle ordering changes.\n\nReferences\n\nM. Chalela, E. Sillero, L. Pereyra, M.A. Garcia, J.B. Cabral, M. Lares, M. Merchán. \"GriSPy: A Python package for fixed-radius nearest neighbors search\". In: Astronomy and Computing 34 (2021). doi: 10.1016/j.ascom.2020.100443\nMarkus Ihmsen, Nadir Akinci, Markus Becker, Matthias Teschner. \"A Parallel SPH Implementation on Multi-Core CPUs\". In: Computer Graphics Forum 30.1 (2011), pages 99–112. doi: 10.1111/J.1467-8659.2010.01832.X\n\n\n\n\n\n","category":"type"},{"location":"reference/#PointNeighbors.PrecomputedNeighborhoodSearch","page":"API reference","title":"PointNeighbors.PrecomputedNeighborhoodSearch","text":"PrecomputedNeighborhoodSearch{NDIMS}(search_radius, n_particles;\n                                     periodic_box_min_corner = nothing,\n                                     periodic_box_max_corner = nothing)\n\nNeighborhood search with precomputed neighbor lists. A list of all neighbors is computed for each particle during initialization and update. This neighborhood search maximizes the performance of neighbor loops at the cost of a much slower update!.\n\nA GridNeighborhoodSearch is used internally to compute the neighbor lists during initialization and update.\n\nArguments\n\nNDIMS:          Number of dimensions.\nsearch_radius:  The uniform search radius.\nn_particles:    Total number of particles.\n\nKeywords\n\nperiodic_box_min_corner:    In order to use a (rectangular) periodic domain, pass the                               coordinates of the domain corner in negative coordinate                               directions.\nperiodic_box_max_corner:    In order to use a (rectangular) periodic domain, pass the                               coordinates of the domain corner in positive coordinate                               directions.\n\n\n\n\n\n","category":"type"},{"location":"reference/#PointNeighbors.TrivialNeighborhoodSearch","page":"API reference","title":"PointNeighbors.TrivialNeighborhoodSearch","text":"TrivialNeighborhoodSearch{NDIMS}(search_radius, eachparticle)\n\nTrivial neighborhood search that simply loops over all particles. The search radius still needs to be passed in order to sort out particles outside the search radius in the internal function for_particle_neighbor, but it's not used in the internal function eachneighbor.\n\nArguments\n\nNDIMS:          Number of dimensions.\nsearch_radius:  The uniform search radius.\neachparticle:   UnitRange of all particle indices. Usually just 1:n_particles.\n\nKeywords\n\nperiodic_box_min_corner:    In order to use a (rectangular) periodic domain, pass the                               coordinates of the domain corner in negative coordinate                               directions.\nperiodic_box_max_corner:    In order to use a (rectangular) periodic domain, pass the                               coordinates of the domain corner in positive coordinate                               directions.\n\n\n\n\n\n","category":"type"},{"location":"reference/#PointNeighbors.initialize!-Tuple{PointNeighbors.AbstractNeighborhoodSearch, Any, Any}","page":"API reference","title":"PointNeighbors.initialize!","text":"initialize!(search::AbstractNeighborhoodSearch, x, y)\n\nInitialize a neighborhood search with the two coordinate arrays x and y.\n\nIn general, the purpose of a neighborhood search is to find for one point in x all points in y whose distances to that point are smaller than the search radius. x and y are expected to be matrices, where the i-th column contains the coordinates of point i. Note that x and y can be identical.\n\nSee also update!.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PointNeighbors.update!-Tuple{PointNeighbors.AbstractNeighborhoodSearch, Any, Any}","page":"API reference","title":"PointNeighbors.update!","text":"update!(search::AbstractNeighborhoodSearch, x, y; particles_moving = (true, true))\n\nUpdate an already initialized neighborhood search with the two coordinate arrays x and y.\n\nLike initialize!, but reusing the existing data structures of the already initialized neighborhood search. When the points only moved a small distance since the last update! or initialize!, this is significantly faster than initialize!.\n\nNot all implementations support incremental updates. If incremental updates are not possible for an implementation, update! will fall back to a regular initialize!.\n\nSome neighborhood searches might not need to update when only x changed since the last update! or initialize! and y did not change. Pass particles_moving = (true, false) in this case to avoid unnecessary updates. The first flag in particles_moving indicates if points in x are moving. The second flag indicates if points in y are moving.\n\nSee also initialize!.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PointNeighbors.@threaded-Tuple{Any}","page":"API reference","title":"PointNeighbors.@threaded","text":"@threaded for ... end\n\nSemantically the same as Threads.@threads when iterating over a AbstractUnitRange but without guarantee that the underlying implementation uses Threads.@threads or works for more general for loops. In particular, there may be an additional check whether only one thread is used to reduce the overhead of serial execution or the underlying threading capabilities might be provided by other packages such as Polyester.jl.\n\nwarn: Warn\nThis macro does not necessarily work for general for loops. For example, it does not necessarily support general iterables such as eachline(filename).\n\nSome discussion can be found at https://discourse.julialang.org/t/overhead-of-threads-threads/53964 and https://discourse.julialang.org/t/threads-threads-with-one-thread-how-to-remove-the-overhead/58435.\n\nCopied from Trixi.jl.\n\n\n\n\n\n","category":"macro"},{"location":"","page":"Home","title":"Home","text":"EditURL = \"https://github.com/trixi-framework/PointNeighbors.jl/blob/main/README.md\"","category":"page"},{"location":"#PointNeighbors.jl","page":"Home","title":"PointNeighbors.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"(Image: Docs-stable) (Image: Docs-dev) (Image: Slack) (Image: Youtube) (Image: Build Status) (Image: Codecov) (Image: SciML Code Style) (Image: License: MIT)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Work in Progress!","category":"page"}]
}
