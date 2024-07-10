# PointNeighbors.jl

[![Docs-stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://trixi-framework.github.io/PointNeighbors.jl/stable)
[![Docs-dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://trixi-framework.github.io/PointNeighbors.jl/dev)
[![Slack](https://img.shields.io/badge/chat-slack-e01e5a)](https://join.slack.com/t/trixi-framework/shared_invite/zt-sgkc6ppw-6OXJqZAD5SPjBYqLd8MU~g)
[![Youtube](https://img.shields.io/youtube/channel/views/UCpd92vU2HjjTPup-AIN0pkg?style=social)](https://www.youtube.com/@trixi-framework)
[![Build Status](https://github.com/trixi-framework/PointNeighbors.jl/workflows/CI/badge.svg)](https://github.com/trixi-framework/PointNeighbors.jl/actions?query=workflow%3ACI)
[![Codecov](https://codecov.io/gh/trixi-framework/PointNeighbors.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/trixi-framework/PointNeighbors.jl)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)
[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/license/mit/)
[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.12702157.svg)](https://zenodo.org/doi/10.5281/zenodo.12702157)

**PointNeighbors.jl** is a package for neighborhood search with fixed search radius in
1D, 2D and 3D point clouds.

## Features

- Several implementations of neighborhood search with fixed search radius
- Focus on fast incremental updates to be usable for particle-based simulations with
  frequent updates
- Designed as a "playground" to easily switch between different implementations and data
  structures
- Common API over all implementations
- Extensive benchmark suite to study different implementations (work in progress)
- GPU compatibility (work in progress)

| Implementation  | Description | Features | Query | Update | GPU-compatible |
| ------------- | ------------- | --- | :--: | :--: | :--: |
| `GridNeighborhoodSearch` with `DictionaryCellList` | Grid-based NHS with Julia `Dict` backend | Infinite domain | Fast | Fast | ❌ |
| `GridNeighborhoodSearch` with `FullGridCellList` | Grid-based NHS allocating all cells of the domain | Finite domain, but efficient memory layout for densely filled domain. | Faster | Fastest | ✅ |
| `PrecomputedNeighborhoodSearch` | Precompute neighbor lists | Best for [TLSPH](https://trixi-framework.github.io/TrixiParticles.jl/stable/systems/total_lagrangian_sph/) without NHS updates. Not suitable for updates in every time step. | Fastest | Very slow | ❌ |

## Benchmarks

The following benchmarks were conducted on an AMD Ryzen Threadripper 3990X using 128 threads.

Benchmark of a single force computation step of a Weakly Compressible SPH (WCSPH) simulation:
![wcsph](https://github.com/trixi-framework/PointNeighbors.jl/assets/44124897/ad5c378b-9ce2-4e6f-91dc-1e0da379b91f)

Benchmark of an incremental update similar to a WCSPH simulation (note the log scale):
![update](https://github.com/trixi-framework/PointNeighbors.jl/assets/44124897/71eac5c9-6aa5-4267-bc0b-4057c89f8b12)

Benchmark of a full right-hand side evaluation of a WCSPH simulation (note the log scale):
![rhs](https://github.com/trixi-framework/PointNeighbors.jl/assets/44124897/ac328a96-1b9f-4319-a785-dce9d862fd70)


## Packages using PointNeighbors.jl

- [TrixiParticles.jl](https://github.com/trixi-framework/TrixiParticles.jl)
- [Peridynamics.jl](https://github.com/kaipartmann/Peridynamics.jl)

If you're using PointNeighbors.jl in your package, please feel free to open a PR adding it
to this list.


## Cite Us

If you use PointNeighbors.jl in your own research or write a paper using results obtained
with the help of PointNeighbors.jl, please cite it as
```bibtex
@misc{pointneighbors,
  title={{P}oint{N}eighbors.jl: {N}eighborhood search with fixed search radius in {J}ulia},
  author={Erik Faulhaber and Niklas Neher and Sven Berger and
          Michael Schlottke-Lakemper and Gregor Gassner},
  year={2024},
  howpublished={\url{https://github.com/trixi-framework/PointNeighbors.jl}},
  doi={10.5281/zenodo.12702157}
}
```
