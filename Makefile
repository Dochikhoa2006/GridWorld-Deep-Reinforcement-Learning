PYTHON ?= python3
VENV ?= .venv
PYTHON_BIN := $(VENV)/bin/python

.DEFAULT_GOAL := help

.PHONY: help venv install-dev format format-check lint test test-unit test-integration coverage-html build smoke pre-commit check

help: ## Show the available development commands.
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z_-]+:.*## / {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

venv: $(PYTHON_BIN) ## Create the local virtual environment.

$(PYTHON_BIN):
	$(PYTHON) -m venv $(VENV)

install-dev: venv ## Install the package and pinned development dependencies.
	$(PYTHON_BIN) -m pip install -r requirements-dev.txt

format: ## Apply Ruff lint fixes and formatting.
	$(PYTHON_BIN) -m ruff check --fix .
	$(PYTHON_BIN) -m ruff format .

format-check: ## Check formatting without changing files.
	$(PYTHON_BIN) -m ruff format --check .

lint: ## Run the Ruff linter.
	$(PYTHON_BIN) -m ruff check .

test: ## Run all tests with the enforced coverage threshold.
	$(PYTHON_BIN) -m pytest

test-unit: ## Run the fast unit-test suite.
	$(PYTHON_BIN) -m pytest tests/unit --no-cov

test-integration: ## Run the integration-test suite.
	$(PYTHON_BIN) -m pytest tests/integration --no-cov

coverage-html: ## Run all tests and create an HTML coverage report.
	$(PYTHON_BIN) -m pytest --cov-report=html

build: ## Build wheel and source distributions through PEP 517.
	$(PYTHON_BIN) -m build

smoke: ## Verify both installed CLI entry points.
	$(PYTHON_BIN) -m gridworld_rl --help
	$(VENV)/bin/gridworld-rl --help

pre-commit: ## Run every configured pre-commit hook.
	$(PYTHON_BIN) -m pre_commit run --all-files

check: format-check lint test build smoke ## Run the local release-readiness checks.
