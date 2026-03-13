Cloud GitHub Actions are intentionally disabled during the research-grade `v2` overhaul.

All workflow definitions have been renamed from `*.yml` to `*.yml.disabled` so pushes, pull requests, tags, schedules, and comments do not trigger GitHub-hosted runners.

Local equivalents remain available through:

- `Makefile`
- `docker-compose.yml`
- direct Julia entrypoints in `test/`, `docs/`, and `validation/`

To re-enable GitHub Actions after the overhaul, rename the relevant files back to `*.yml` and review the trigger policy before pushing.
