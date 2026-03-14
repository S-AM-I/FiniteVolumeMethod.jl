# Makefile — Local CI for FiniteVolumeMethod.jl
#
# Requires: Docker Desktop (with >= 12 GB memory allocated)
#
# First run:  make ci-build   (15-30 min, downloads + precompiles all deps)
# Then:       make ci-test    (uses cached depot volume)

.PHONY: help ci-build ci-test ci-test-file ci-evidence ci-format ci-format-fix \
        ci-docs ci-docs-ci ci-report ci-bundles ci-release-outputs ci-repl ci-all \
        ci-fast ci-smoke ci-full-evidence ci-release-audit ci-clean ci-depot-clean

COMPOSE := docker-compose

help: ## Show this help
	@echo "FiniteVolumeMethod.jl — Local CI targets"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Tip: Docker Desktop → Preferences → Resources → set memory to 12-16 GB"

ci-build: ## Build base image (run after Project.toml changes)
	$(COMPOSE) build base

ci-test: ## Run full test suite (mirrors .github/workflows/CI.yml.disabled)
	$(COMPOSE) run --rm test

ci-test-file: ## Run single test file (TEST_FILE=test/geometry.jl make ci-test-file)
	$(COMPOSE) run --rm test-file

ci-fast: ## Run the fast API/interop lane
	CI_LANE=fast-api-interop $(COMPOSE) run --rm lane

ci-smoke: ## Run the scientific smoke lane for stable solver families
	CI_LANE=scientific-smoke $(COMPOSE) run --rm lane

ci-full-evidence: ## Run the full scientific evidence lane
	CI_LANE=full-evidence $(COMPOSE) run --rm lane

ci-release-audit: ## Run the release-audit lane with stable release outputs
	CI_LANE=release-audit $(COMPOSE) run --rm lane

ci-evidence: ## Run curated scientific-evidence suite (mirrors CI scientific-evidence lane)
	$(COMPOSE) run --rm evidence

ci-format: ## Check Runic formatting (mirrors .github/workflows/FormatCheck.yml.disabled)
	$(COMPOSE) run --rm format

ci-format-fix: ## Auto-fix Runic formatting
	$(COMPOSE) run --rm format-fix

ci-docs: ## Build docs with executed examples (slow)
	$(COMPOSE) run --rm docs

ci-docs-ci: ## Build docs with the curated CI subset of executed examples
	$(COMPOSE) run --rm docs-ci

ci-report: ## Generate executed validation report with evidence summaries
	$(COMPOSE) run --rm report

ci-bundles: ## Build reproduction bundles for all evidence-bearing solver families
	$(COMPOSE) run --rm bundles

ci-release-outputs: ## Build release-style outputs (summaries + bundles + report + index)
	$(COMPOSE) run --rm release-outputs

ci-repl: ## Interactive Julia REPL in container
	$(COMPOSE) run --rm repl

ci-all: ci-format ci-fast ci-smoke ci-docs-ci ci-full-evidence ci-release-audit ## Run the full local CI lane stack

ci-clean: ## Remove containers (keeps depot volume)
	$(COMPOSE) down --remove-orphans

ci-depot-clean: ## Remove depot volume (forces full re-precompile)
	$(COMPOSE) down -v --remove-orphans
