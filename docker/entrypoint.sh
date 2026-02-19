#!/bin/bash
# Entrypoint for local CI containers.
# Removes stale Manifest.toml files (may be from a different Julia version)
# and regenerates them. Since the depot has all packages cached, this is fast.
set -e

# Remove host Manifests that may be from a different Julia version
rm -f /workspace/Manifest.toml
rm -f /workspace/test/Manifest.toml
rm -f /workspace/docs/Manifest.toml

exec "$@"
