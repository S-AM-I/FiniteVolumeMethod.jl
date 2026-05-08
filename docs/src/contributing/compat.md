# Julia version & dependency compatibility policy

This page documents the policy that the three sister packages
(CRUDApplication, FiniteVolumeMethod, NuclearWaterChemistry) follow for
Julia version support and `[compat]` ranges. The intent is to match SciML
ecosystem conventions so that users importing the SciML stack alongside
this code don't run into resolver conflicts.

## Julia version support

- The packages target the current **Julia LTS** (`1.10`) as the floor.
  This is the lowest version that resolves cleanly against the SciML
  bundle (Catalyst, ModelingToolkit, OrdinaryDiffEq, SciMLBase).
- The packages are continuously tested against **Julia LTS** and
  **current stable** (`1`). When a new LTS ships, the floor is bumped on
  a six-month horizon and the old LTS is dropped from the matrix.
- **Nightly is *not* on the scientific contract.** It runs in optional /
  manual lanes only. A failure on nightly never blocks a PR or release;
  a failure on LTS or stable does.

## Dependency `[compat]` ranges

- Every direct entry in `[deps]` and `[weakdeps]` must have a corresponding
  `[compat]` line. `Pkg.test()` will warn otherwise; CI flags it.
- Use **caret ranges** by default (`X.Y` ≡ `>= X.Y, < (X+1).0`) for stable
  dependencies. Caret is the SciML default: it allows minor-version updates
  (which the SemVer contract says are non-breaking) without forcing every
  consumer to bump a `Project.toml` whenever an upstream cuts a routine
  patch.
- Span breaking-change boundaries explicitly when the package is
  cross-compatible: `"3, 4"` (any 3.x or 4.x), not `"3"`. Examples:
  `Krylov = "0.9, 0.10"`, `LinearSolve = "2, 3"`, `Catalyst = "14, 15"`.
- Stdlib `[compat]` entries pin to the Julia floor (`"1.10"`). They
  document intent; resolution doesn't require them.
- For the unregistered cx-xd / S-AM-I dependencies (FiniteVolumeMethod,
  NuclearWaterChemistry, CRUDApplication itself), do **not** add `[compat]`
  entries — they conflict with `Pkg.develop` workflows. The cross-repo
  pinning lives in `docs/Project.toml` instead, where `Pkg.develop` is
  explicit.

## Adding a new dependency

1. Open a PR that adds the dep + a `[compat]` line.
2. Justify the addition in the commit body: what does it unlock?
3. Confirm `Pkg.test()` passes locally.
4. CI's `unit-interop` lane will exercise the matrix.

## Bumping a `[compat]` upper bound

When an upstream releases a major version:

1. Run `Pkg.update`; verify nothing in `Pkg.test()` regresses.
2. Extend the range additively: `"3"` → `"3, 4"`. Don't drop the old
   bound until at least one release cycle has passed and the matrix has
   exercised both.
3. Move the floor forward only if (a) the old version actively breaks
   compatibility with another required dep, or (b) the package has
   already taken a hard dep on a feature added in the new version.

## Re-evaluation triggers

Revisit this policy if:

- The Julia LTS cadence changes (currently 12-month LTS, 6-month stable).
- The SciML stack adopts a stricter SemVer contract (e.g., dropping
  caret-by-default).
- The cross-repo pin-via-`Pkg.develop` workflow is replaced by a
  registered package release. At that point, `[compat]` for FVM/NWC
  becomes meaningful.
