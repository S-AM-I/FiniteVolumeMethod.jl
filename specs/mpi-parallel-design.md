---
date: 2026-04-07
---

# Phase 6: MPI Parallelism

**Status**: Design
**Depends on**: Phase 0, Phase 1, Phase 5

## Goal

Add distributed-memory parallelism via a package extension triggered by `using MPI, PartitionedArrays`. Provides distributed mesh with ghost cells, non-blocking halo exchange, and parallel SIMPLE/PISO solver. No changes to core solver code.

## Architecture

All MPI code in `ext/FVMMPIExt/`. The extension wraps `UnstructuredFVMMesh` with ghost cells and communication patterns. Existing Phase 0 operators work unchanged on the local submesh — the extension inserts halo exchanges before gradient/convection operators and uses PartitionedArrays for distributed linear solves.

## Files

All in `ext/FVMMPIExt/`:
- `FVMMPIExt.jl` — entry point
- `distributed_mesh.jl` — DistributedFVMMesh, HaloPattern
- `halo_exchange.jl` — non-blocking MPI send/recv
- `partitioning.jl` — distribute_mesh from global mesh
- `distributed_fields.jl` — distributed field wrappers
- `distributed_solve.jl` — parallel SIMPLE with halo exchanges

## Types

DistributedFVMMesh wraps local submesh with ghost bookkeeping. HaloPattern stores send/recv index maps per neighbor rank. All types defined in the extension module.

## Weak Dependencies

`MPI.jl` + `PartitionedArrays.jl` in `[weakdeps]`. Optional `Metis.jl` for graph partitioning.

## Export (from extension)

distribute_mesh, halo_exchange!, DistributedFVMMesh, HaloPattern, solve_simple_distributed
