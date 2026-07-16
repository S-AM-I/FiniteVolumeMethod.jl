---
tags: [repo/FiniteVolumeMethod.jl, validation]
---

# Validation index

Narrative V&V documentation for FiniteVolumeMethod.jl. The **executable** V&V scripts live in the repo at `validation/*.jl` and `validation/*.toml` — those are the source of truth for what is tested. The files below are the *plans* and *release-process* documents.

## V&V

The narrative plan and research notes moved to `docs/research/` (2026-07-16) so this
directory holds only the load-bearing manifest + runner infrastructure:

- [docs/research/vv-plan.md](../docs/research/vv-plan.md) — primary V&V plan and implementation history
- [docs/research/vv-research-cfd.md](../docs/research/vv-research-cfd.md) — upstream research dump: general CFD V&V (ASME V&V 20, MMS, GCI, classical benchmarks)
- [docs/research/vv-research-mhd-relativistic.md](../docs/research/vv-research-mhd-relativistic.md) — upstream research dump: relativistic and Newtonian MHD V&V

## Release

- [[release-checklist]] — pre-release checklist

## How this maps to executable scripts (in repo)

| Doc | Script(s) it informs |
|-----|---------------------|
| vv-plan | `reference_artifacts.jl`, `evidence_capture.jl`, `evidence_runner.jl` |
| Performance baselines | `performance_baselines.jl`, `performance_baselines.toml`, `performance_calibration.jl` |
| Reproducibility | `reproducibility.jl`, `manifest.jl`, `manifest.toml` |
| Release packaging | `release_packaging.jl`, `release_audit.jl`, `summary_replay.jl` |
| Backend parity | `backend_parity.jl` |
| Project integrity | `project_integrity.jl` |
| Report generation | `generate_report.jl` |
