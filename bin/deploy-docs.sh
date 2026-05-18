#!/usr/bin/env bash
# Build FiniteVolumeMethod.jl docs locally and deploy to the `fvm` Cloudflare Worker.
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
  -e 's|</head>|<style>.VPSwitchAppearance,.VPNavBarAppearance,.VPNavScreenAppearance{display:none!important}html.dark{color-scheme:light!important}html.dark *{color-scheme:light!important}.VPNavBar .content-body{overflow-x:clip;overflow-y:visible;flex-wrap:nowrap;white-space:nowrap}.VPNavBar .VPNavBarMenu{overflow-x:auto;overflow-y:hidden;flex-wrap:nowrap;white-space:nowrap;-webkit-overflow-scrolling:touch}.VPNavBar .VPNavBarMenu:has(.VPFlyout:hover,button[aria-expanded="true"]){overflow:visible}.VPNavBar .VPNavBarMenu::-webkit-scrollbar{height:4px}.VPFlyout .menu,.VPMenu{z-index:100}</style></head>|'

cd "$REPO"
echo "==> deploying to Worker '$PROJECT'"
# wrangler.jsonc at repo root declares name=fvm and assets.directory=./docs/build
npx wrangler deploy
