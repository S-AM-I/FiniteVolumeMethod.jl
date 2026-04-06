# Combustion & Species Transport Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add multi-species transport with EDM reaction model and heat release coupling to the incompressible thermal solver.

**Architecture:** Four files in `src/combustion/`. Species Y_i solved as scalar transport (convection + diffusion + reaction source). EDM computes reaction rates from turbulence mixing time scale. Heat release feeds into energy equation. Wrapper solver combines flow + turbulence + species + energy.

**Tech Stack:** Phase 0 operators, Phase 1 incompressible, Phase 2a turbulence (k/ε for EDM), Phase 3 energy.

---

## Tasks

### Task 1: Create all 4 source files
- `src/combustion/types.jl`, `species_transport.jl`, `edm.jl`, `solvers.jl`

### Task 2: Wire into module + exports

### Task 3: Write tests + register manifest

See spec at `docs/superpowers/specs/2026-04-06-combustion-design.md` for complete code specifications.
