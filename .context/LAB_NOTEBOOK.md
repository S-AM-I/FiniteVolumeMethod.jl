# LAB NOTEBOOK — FiniteVolumeMethod.jl
<!-- Chronological record of experiments, parameter choices, and FAILED attempts. -->
<!-- The most important entries are the ones that DIDN'T work — they prevent retrying dead ends. -->

## Entries

### 2026-02-17 — CI/CD Assessment and Julia Version Matrix
**Goal:** Understand what each GitHub Actions workflow does, whether the test matrix aligns with SciML best practices, and improve the setup.
**Method:** Read all 4 workflow files (`CI.yml`, `FormatCheck.yml`, `TagBot.yml`, `DocCleanup.yml`). Compared matrix configuration against SciML conventions. Assessed docs/test dependency structure.
**Result:**
- Setup is already well-aligned with SciML practices (LTS + stable + nightly, `continue-on-error` on nightly, docs deploy in parallel).
- One issue: Julia version was hardcoded to `'1.11'` instead of `'1'` (auto-latest). Changed it.
- `pre` job failure (red X) is expected and non-blocking due to `continue-on-error: true`.
- Docs and tests run in parallel (no `needs:` dependency), so docs aren't blocked by slow tests.
- No CompatHelper workflow, but Dependabot serves a similar purpose — acceptable.
**Conclusion:** Only change needed was `'1.11'` → `'1'` in CI.yml. Everything else is correct.
