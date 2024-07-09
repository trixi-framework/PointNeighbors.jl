# PointNeighbors.jl

[![Docs-stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://trixi-framework.github.io/PointNeighbors.jl/stable)
[![Docs-dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://trixi-framework.github.io/PointNeighbors.jl/dev)
[![Slack](https://img.shields.io/badge/chat-slack-e01e5a)](https://join.slack.com/t/trixi-framework/shared_invite/zt-sgkc6ppw-6OXJqZAD5SPjBYqLd8MU~g)
[![Youtube](https://img.shields.io/youtube/channel/views/UCpd92vU2HjjTPup-AIN0pkg?style=social)](https://www.youtube.com/@trixi-framework)
[![Build Status](https://github.com/trixi-framework/PointNeighbors.jl/workflows/CI/badge.svg)](https://github.com/trixi-framework/PointNeighbors.jl/actions?query=workflow%3ACI)
[![Codecov](https://codecov.io/gh/trixi-framework/PointNeighbors.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/trixi-framework/PointNeighbors.jl)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)
[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/license/mit/)

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

## Benchmarks

 

## Packages using PointNeighbors.jl

- [TrixiParticles.jl](https://github.com/trixi-framework/TrixiParticles.jl)
- [Peridynamics.jl](https://github.com/kaipartmann/Peridynamics.jl)

If you're using PointNeighbors.jl in your package, please feel free to open a PR adding it
to this list.
