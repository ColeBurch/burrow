SHELL := /bin/sh

MODULES := ai agent baseAgent
PACKAGES := ./ai/... ./agent/... ./baseAgent/...
REMOTE ?= origin
RELEASE_BRANCH ?= main

.DEFAULT_GOAL := help

.PHONY: help build test test-race coverage vet fmt fmt-check tidy download vendor \
	generate generate-models check clean versions fetch-tags \
	bump bump-patch bump-minor bump-major \
	tag tag-patch tag-minor tag-major publish

# This macro sets ai_version, agent_version, and base_agent_version.
define compute_next_versions
next_version() { \
	module="$$1"; \
	latest=$$(git tag --list "$$module/v*" --sort=-version:refname | \
		grep -E "^$$module/v[0-9]+\.[0-9]+\.[0-9]+$$" | head -n 1); \
	if [ -n "$$latest" ]; then \
		version=$${latest#$$module/v}; \
	else \
		version=0.0.0; \
	fi; \
	major=$$(printf '%s\n' "$$version" | awk -F. '{print $$1}'); \
	minor=$$(printf '%s\n' "$$version" | awk -F. '{print $$2}'); \
	patch=$$(printf '%s\n' "$$version" | awk -F. '{print $$3}'); \
	case "$(BUMP)" in \
		patch) patch=$$((patch + 1)) ;; \
		minor) minor=$$((minor + 1)); patch=0 ;; \
		major) major=$$((major + 1)); minor=0; patch=0 ;; \
		*) printf 'BUMP must be patch, minor, or major.\n' >&2; return 2 ;; \
	esac; \
	printf '%s.%s.%s\n' "$$major" "$$minor" "$$patch"; \
}; \
ai_version=$$(next_version ai); \
agent_version=$$(next_version agent); \
base_agent_version=$$(next_version baseAgent)
endef

# Go requires a /vN module-path suffix for major versions after v1.
define validate_major_paths
validate_major_path() { \
	module="$$1"; \
	version="$$2"; \
	major=$${version%%.*}; \
	if [ "$$major" -lt 2 ]; then return 0; fi; \
	module_path=$$(awk '$$1 == "module" { print $$2; exit }' "$$module/go.mod"); \
	case "$$module_path" in \
		*/v$$major) return 0 ;; \
	esac; \
	printf '%s v%s requires a module path that ends in /v%s.\n' "$$module" "$$version" "$$major" >&2; \
	printf 'Update the module paths and imports before this release.\n' >&2; \
	return 1; \
}; \
validate_major_path ai "$$ai_version"; \
validate_major_path agent "$$agent_version"; \
validate_major_path baseAgent "$$base_agent_version"
endef

help:
	@printf '%s\n' \
		'Common targets:' \
		'  make build          Compile all modules.' \
		'  make test           Run all tests.' \
		'  make test-race      Run all tests with the race detector.' \
		'  make coverage       Write coverage.out and show the coverage summary.' \
		'  make vet            Run go vet for all modules.' \
		'  make fmt            Format all Go files.' \
		'  make fmt-check      Report Go files that need formatting.' \
		'  make tidy           Tidy each Go module.' \
		'  make download       Download dependencies for each Go module.' \
		'  make vendor         Synchronize the workspace vendor directory.' \
		'  make generate       Generate ai/models.generated.go.' \
		'  make check          Run formatting, vet, and test checks.' \
		'  make versions       Show the latest module tags.' \
		'' \
		'Release targets:' \
		'  make bump-patch     Prepare internal requirements for patch tags.' \
		'  make bump-minor     Prepare internal requirements for minor tags.' \
		'  make bump-major     Prepare internal requirements for major tags.' \
		'  make tag-patch      Check and tag a clean release commit.' \
		'  make tag-minor      Check and tag a clean release commit.' \
		'  make tag-major      Check and tag a clean release commit.' \
		'  make publish        Push the release commit and its module tags.' \
		'' \
		'Release flow:' \
		'  1. Run make bump-patch, bump-minor, or bump-major.' \
		'  2. Commit all release changes.' \
		'  3. Run the matching tag target.' \
		'  4. Run make publish.'

build:
	go build $(PACKAGES)

test:
	go test $(PACKAGES)

test-race:
	go test -race $(PACKAGES)

coverage:
	go test -coverprofile=coverage.out $(PACKAGES)
	go tool cover -func=coverage.out

vet:
	go vet $(PACKAGES)

fmt:
	@gofmt -w $$(find . -type f -name '*.go' -not -path './vendor/*')

fmt-check:
	@files=$$(gofmt -l $$(find . -type f -name '*.go' -not -path './vendor/*')); \
	if [ -n "$$files" ]; then \
		printf 'These files need formatting:\n%s\n' "$$files"; \
		exit 1; \
	fi

tidy:
	@for module in . $(MODULES); do \
		printf 'Tidying %s\n' "$$module"; \
		(cd "$$module" && go mod tidy); \
	done

download:
	@for module in . $(MODULES); do \
		printf 'Downloading dependencies for %s\n' "$$module"; \
		(cd "$$module" && go mod download); \
	done

vendor:
	go work vendor

generate: generate-models

generate-models:
	go run ./scripts/generate-models.go

check: fmt-check
	@git diff --check
	@$(MAKE) --no-print-directory vet
	@$(MAKE) --no-print-directory test

clean:
	@rm -f coverage.out
	@go clean -testcache

versions:
	@for module in $(MODULES); do \
		tag=$$(git tag --list "$$module/v*" --sort=-version:refname | \
			grep -E "^$$module/v[0-9]+\.[0-9]+\.[0-9]+$$" | head -n 1); \
		if [ -z "$$tag" ]; then tag='not released'; fi; \
		printf '%-12s %s\n' "$$module" "$$tag"; \
	done

fetch-tags:
	git fetch $(REMOTE) --tags

bump-patch:
	@$(MAKE) --no-print-directory bump BUMP=patch

bump-minor:
	@$(MAKE) --no-print-directory bump BUMP=minor

bump-major:
	@$(MAKE) --no-print-directory bump BUMP=major

bump:
	@set -eu; \
	$(compute_next_versions); \
	$(validate_major_paths); \
	ai_path=$$(awk '$$1 == "module" { print $$2; exit }' ai/go.mod); \
	agent_path=$$(awk '$$1 == "module" { print $$2; exit }' agent/go.mod); \
	(cd agent && go mod edit -require="$$ai_path@v$$ai_version"); \
	(cd baseAgent && go mod edit \
		-require="$$ai_path@v$$ai_version" \
		-require="$$agent_path@v$$agent_version"); \
	if [ -f vendor/modules.txt ]; then go work vendor; fi; \
	printf 'Prepared internal module requirements.\n\n'; \
	printf 'Planned tags:\n'; \
	printf '  ai/v%s\n' "$$ai_version"; \
	printf '  agent/v%s\n' "$$agent_version"; \
	printf '  baseAgent/v%s\n' "$$base_agent_version"; \
	printf '\nCommit the go.mod changes before you create the tags.\n'

tag-patch:
	@$(MAKE) --no-print-directory tag BUMP=patch

tag-minor:
	@$(MAKE) --no-print-directory tag BUMP=minor

tag-major:
	@$(MAKE) --no-print-directory tag BUMP=major

tag: check
	@set -eu; \
	if [ -n "$$(git status --porcelain)" ]; then \
		printf 'The worktree is not clean. Commit all release changes first.\n' >&2; \
		exit 1; \
	fi; \
	branch=$$(git rev-parse --abbrev-ref HEAD); \
	if [ "$$branch" != "$(RELEASE_BRANCH)" ]; then \
		printf 'The release branch must be %s. Current branch: %s\n' "$(RELEASE_BRANCH)" "$$branch" >&2; \
		exit 1; \
	fi; \
	$(compute_next_versions); \
	$(validate_major_paths); \
	ai_path=$$(awk '$$1 == "module" { print $$2; exit }' ai/go.mod); \
	agent_path=$$(awk '$$1 == "module" { print $$2; exit }' agent/go.mod); \
	agent_ai_version=$$(cd agent && GOWORK=off go list -m -f '{{.Version}}' "$$ai_path"); \
	base_ai_version=$$(cd baseAgent && GOWORK=off go list -m -f '{{.Version}}' "$$ai_path"); \
	base_agent_version_required=$$(cd baseAgent && GOWORK=off go list -m -f '{{.Version}}' "$$agent_path"); \
	if [ "$$agent_ai_version" != "v$$ai_version" ]; then \
		printf 'agent/go.mod requires %s. Expected v%s.\n' "$$agent_ai_version" "$$ai_version" >&2; \
		exit 1; \
	fi; \
	if [ "$$base_ai_version" != "v$$ai_version" ]; then \
		printf 'baseAgent/go.mod requires %s. Expected v%s.\n' "$$base_ai_version" "$$ai_version" >&2; \
		exit 1; \
	fi; \
	if [ "$$base_agent_version_required" != "v$$agent_version" ]; then \
		printf 'baseAgent/go.mod requires %s. Expected v%s.\n' "$$base_agent_version_required" "$$agent_version" >&2; \
		exit 1; \
	fi; \
	for tag in "ai/v$$ai_version" "agent/v$$agent_version" "baseAgent/v$$base_agent_version"; do \
		if git rev-parse -q --verify "refs/tags/$$tag" >/dev/null; then \
			printf 'Tag already exists: %s\n' "$$tag" >&2; \
			exit 1; \
		fi; \
	done; \
	git tag "ai/v$$ai_version"; \
	git tag "agent/v$$agent_version"; \
	git tag "baseAgent/v$$base_agent_version"; \
	printf 'Created tags:\n'; \
	printf '  ai/v%s\n' "$$ai_version"; \
	printf '  agent/v%s\n' "$$agent_version"; \
	printf '  baseAgent/v%s\n' "$$base_agent_version"

publish:
	@set -eu; \
	if [ -n "$$(git status --porcelain)" ]; then \
		printf 'The worktree is not clean. Commit all release changes first.\n' >&2; \
		exit 1; \
	fi; \
	branch=$$(git rev-parse --abbrev-ref HEAD); \
	if [ "$$branch" != "$(RELEASE_BRANCH)" ]; then \
		printf 'The release branch must be %s. Current branch: %s\n' "$(RELEASE_BRANCH)" "$$branch" >&2; \
		exit 1; \
	fi; \
	tags=''; \
	for module in $(MODULES); do \
		tag=$$(git tag --points-at HEAD --list "$$module/v*" --sort=-version:refname | \
			grep -E "^$$module/v[0-9]+\.[0-9]+\.[0-9]+$$" | head -n 1); \
		if [ -z "$$tag" ]; then \
			printf 'HEAD does not have a release tag for %s.\n' "$$module" >&2; \
			exit 1; \
		fi; \
		tags="$$tags $$tag"; \
	done; \
	printf 'Publishing%s\n' "$$tags"; \
	git push --atomic "$(REMOTE)" "$(RELEASE_BRANCH)" $$tags
