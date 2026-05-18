#!/usr/bin/env bash
# Build FiniteVolumeMethod.jl docs locally and deploy to the `fvm` Cloudflare Pages project.
#
# Usage: bin/deploy-docs.sh

set -euo pipefail
PROJECT=fvm
REPO=$(cd "$(dirname "$0")/.." && pwd)

cd "$REPO"
echo "==> building $PROJECT docs"
CI=true julia --project=docs docs/make.jl

cd docs/build
echo "==> post-processing: force light theme, hide toggle, fix nav dropdown clipping"
find . -name '*.html' -print0 | xargs -0 sed -i '' \
  -e 's|<head>|<head><meta name="color-scheme" content="light only">|' \
  -e 's|<script id="check-dark-mode">|<script>localStorage.setItem("vitepress-theme-appearance","light");document.documentElement.classList.remove("dark");</script><script id="check-dark-mode">|' \
  -e 's|</head>|<style>.VPSwitchAppearance,.VPNavBarAppearance,.VPNavScreenAppearance{display:none!important}html.dark{color-scheme:light!important}html.dark *{color-scheme:light!important}.VPNavBar .content-body,.VPNavBar nav,.VPNavBarMenu{overflow-x:clip;overflow-y:visible;flex-wrap:nowrap;white-space:nowrap}.VPFlyout .menu,.VPMenu{z-index:100}</style></head>|'

echo "==> deploying to project '$PROJECT'"
wrangler pages deploy . --project-name="$PROJECT" --branch main --commit-dirty=true
