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

cd docs/build/1
echo "==> post-processing: default to light, scrollable nav"
find . -name '*.html' -print0 | xargs -0 sed -i '' \
  -e 's|<script id="check-dark-mode">|<script>if(!localStorage.getItem("vitepress-theme-appearance"))localStorage.setItem("vitepress-theme-appearance","light");</script><script id="check-dark-mode">|' \
  -e 's|</head>|<style>.VPNavBar .content-body,.VPNavBar nav,.VPNavBarMenu{overflow-x:auto;flex-wrap:nowrap;white-space:nowrap;-webkit-overflow-scrolling:touch}.VPNavBar .content-body::-webkit-scrollbar,.VPNavBarMenu::-webkit-scrollbar{height:4px}</style></head>|'

echo "==> deploying to project '$PROJECT'"
wrangler pages deploy . --project-name="$PROJECT" --branch main --commit-dirty=true
