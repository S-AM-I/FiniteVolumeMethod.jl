---
title: FiniteVolumeMethod.jl
type: project
status: active
domain: nuclear
tags: [project, project/active]
repos: [github.com/cx-xd/FiniteVolumeMethod.jl]
started:
target:
stack: julia
live:
updated: 2026-06-03
---

> General-purpose Julia PDE solver — parabolic/elliptic on triangular meshes, hyperbolic 1/2/3-D; with MHD, relativistic, AMR, coupling, and a V&V suite.

## Status

43 solver modules, comprehensive V&V suite, Documenter.jl API site. Current capability focus and the gap snapshot live in [[openfoam-gap-analysis]].

## Repos

- [FiniteVolumeMethod.jl](https://github.com/cx-xd/FiniteVolumeMethod.jl) — the library. Open: `repo-open FiniteVolumeMethod.jl`
- **Canonical docs:** the Documenter site (built from `docs/`) is no longer hosted — the CF Pages host was retired; build it locally or browse `docs/src/`. The repo holds code, build configs, and machine-checkable validation artifacts; the vault owns the planning/gap narrative below.

## Now

- [ ]

## Log

-

## Notes

### Companion notes (this folder)

- [[openfoam-gap-analysis]] — feature gap vs OpenFOAM
- `plans/` — per-capability planning + [[plans/open-work|open-work]]
- `specs/` — per-capability design specs
- `validation/` — V&V plans, release checklists, RC status reports

### Related

- [[Reactor]] — uses FVM as its PDE backbone
- [[CRUD]] — thesis using the FVM+Reactor stack
