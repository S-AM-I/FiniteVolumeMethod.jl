#!/usr/bin/env bash
# Install a pre-commit hook that runs Runic on staged Julia files.
#
# Usage: ./scripts/install_hooks.sh
#
# Idempotent: overwrites any existing pre-commit hook with the same logic.

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
hook="${repo_root}/.git/hooks/pre-commit"

cat > "${hook}" <<'HOOK'
#!/usr/bin/env bash
# Pre-commit: Runic format check on staged Julia files.
set -euo pipefail

mapfile -t files < <(git diff --cached --name-only --diff-filter=ACMR | grep -E '\.jl$' || true)
if [[ ${#files[@]} -eq 0 ]]; then
    exit 0
fi

if ! command -v julia >/dev/null 2>&1; then
    echo "pre-commit: julia not on PATH; skipping Runic check" >&2
    exit 0
fi

if ! julia -e 'using Runic' 2>/dev/null; then
    echo "pre-commit: Runic not installed in default environment; install with"
    echo "    julia -e 'using Pkg; Pkg.add(\"Runic\")'"
    exit 1
fi

if ! julia -e 'using Runic; Runic.main(["--check", ARGS...])' "${files[@]}"; then
    echo
    echo "pre-commit: Runic format violations above. Run:"
    echo "    julia -e 'using Runic; Runic.main([\"--inplace\", ARGS...])' ${files[*]}"
    echo "to fix, then re-stage and commit."
    exit 1
fi
HOOK

chmod +x "${hook}"
echo "Installed pre-commit hook: ${hook}"
