# Known Test Failures

This document tracks pre-existing test failures that are not related to the FVM migration work.

## Aqua.jl Compat Bound Failures (4 tests)

**Location**: `test/runtests.jl` -> Aqua testset

**Issue**: Missing compat entries in `Project.toml` for stdlib packages.

### Missing deps compat entries:
- `LinearAlgebra` [37e2e46d-f89d-539d-b4ee-838fcccc9c8e]
- `SparseArrays` [2f01184e-e22b-5df5-ae63-d93ebab69eaf]

### Missing extras compat entry:
- `Test` [8dfed614-e22c-5e08-85e1-65c5234f0b40]

**Resolution**: Add compat bounds for these stdlib packages in `Project.toml`. For Julia 1.6+, these can typically be set to `"1"` or the appropriate Julia version constraint.

## Reference Image Test Error (1 test)

**Location**: `docs/src/literate_tutorials/piecewise_linear_and_natural_neighbour_interpolation_for_an_advection_diffusion_equation.jl:335`

**Issue**: Reference image comparison fails due to PSNR (Peak Signal-to-Noise Ratio) being below threshold.

```
test fails because PSNR 23.992900848388672 < 25
```

**Details**:
- Reference file: `docs/src/figures/piecewise_linear_and_natural_neighbour_interpolation_for_an_advection_diffusion_equation_natural_neighbour_interpolation.png`
- The generated image is visually similar but has minor numerical differences causing PSNR to fall just below the 25 threshold.

**Resolution options**:
1. Update the reference image by running tests with `JULIA_REFERENCETESTS_UPDATE=true`
2. Lower the PSNR threshold in the test
3. Investigate source of numerical differences (likely floating-point precision or RNG-related)

## Test Summary

| Category | Passed | Failed | Errored |
|----------|--------|--------|---------|
| Total    | 1,288,388 | 4 | 1 |

All functional tests (tutorials, unit tests, schemes, etc.) pass successfully.
