# API Reference

Reference documentation is split across two kinds of page.

**Curated interface pages** group a family's public API semantically — meshes,
equations of state, conservation laws, boundary conditions, and so on — with
prose explaining how the pieces fit together. They are the better starting
point when you are learning a solver family:

- [Parabolic Solver Interface](../interface.md)
- [Hyperbolic Solver Interface](../hyperbolic/interface.md)
- [Writing Your Own Solvers](../wyos/overview.md)

Those pages also cover the `Geometry`, `Numerics` and `VertexConditions`
modules, since the types involved are shared.

**Generated module pages** — the ones in this section — cover the modules that
have no curated interface page. Each lists every documented object in the
module, split into:

- **Public API** — names the module exports, or marks `public`. This is the
  supported surface: it follows semantic versioning and is what user code
  should depend on.
- **Internal** — documented implementation detail, listed because it is useful
  when reading or extending the solvers. Not part of the supported surface, and
  may change without a breaking release.

Only objects that carry a docstring appear here, so a name's absence means it is
undocumented rather than non-existent.

| Module | Contents |
|---|---|
| [Collocated](collocated.md) | Collocated incompressible solvers and their composed physics |
| [I/O and Session Tooling](io.md) | Output, diagnostics, checkpointing, dashboard |
| [Experimental](experimental.md) | Quarantined research scaffolds |

## Scope and maturity

Appearing in this reference says nothing about scientific maturity. Whether a
capability is validated to publication grade is recorded separately in the
[Capability Matrix](../capability_matrix.md) and the validation manifest, which
are the authoritative contract. Most of the collocated stack is `experimental`.

The `Experimental` module in particular is research scaffolding: its entry
points warn once per session and are excluded from the package's validation
claims — see [Experimental scope and caveats](../experimental/overview.md).
