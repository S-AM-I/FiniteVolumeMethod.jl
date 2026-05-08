# `theory/` — executable theory pages

Layer-2 documentation: theory pages whose Julia code blocks run on every
render. Pipeline mirrors CRUDApplication.jl and NuclearWaterChemistry.jl
for cross-repo consistency.

## Install Quarto + QuartoNotebookRunner

```bash
# Quarto (single binary — pacman / brew / install.sh)
pacman -S quarto-cli   # or: brew install quarto

# QuartoNotebookRunner (Julia kernel)
julia --project=. -e 'using Pkg; Pkg.add("QuartoNotebookRunner"); Pkg.instantiate()'
```

## Render

```bash
cd theory
quarto render --to gfm
```

Renders to `../docs/src/theory/*.md`, picked up by `docs/make.jl`'s
pages array via the existing DocumenterVitepress flow.

## Status

Skeleton only. The first FVM theory page to migrate will be *parabolic
MMS convergence* — `verification/mms_convergence.md` ports cleanly to
`theory/parabolic_mms.qmd` because `validation/manifest.jl` already
generates the rate table.
