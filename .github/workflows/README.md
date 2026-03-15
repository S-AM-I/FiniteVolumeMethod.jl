Cloud GitHub Actions are intentionally disabled during the research-grade `v2` overhaul.

All workflow definitions have been renamed from `*.yml` to `*.yml.disabled` so pushes, pull requests, tags, schedules, and comments do not trigger GitHub-hosted runners.

Local equivalents remain available through:

- `Makefile`
- `docker-compose.yml`
- direct Julia entrypoints in `test/`, `docs/`, and `validation/`

The staged re-enable criteria and proposed cloud mapping live in `validation/CI_REENABLE_PLAN.md`.

To re-enable GitHub Actions after the overhaul, rename the relevant files back to `*.yml`, apply the trigger policy from `validation/CI_REENABLE_PLAN.md`, and keep docs deployment gated on green scientific and release lanes.
