---
tags: [repo/FiniteVolumeMethod.jl, validation]
---

# Validation index

Narrative V&V documentation for FiniteVolumeMethod.jl. The **executable** V&V scripts live in the repo at `validation/*.jl` and `validation/*.toml` — those are the source of truth for what is tested. The files below are the *plans* and *release-process* documents.

## V&V

- [[vv-plan]] — primary V&V plan and implementation history
- [[vv-research-cfd]] — upstream research dump: general CFD V&V (ASME V&V 20, MMS, GCI, classical benchmarks)
- [[vv-research-mhd-relativistic]] — upstream research dump: relativistic and Newtonian MHD V&V

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
