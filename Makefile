# Makefile for local macOS setup using uv

.PHONY: help setup setup-dev activate run lock clean mac-prereqs install-uv check-uv py312
.DEFAULT_GOAL := help

SHELL := /bin/bash

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sed -E 's/:.*## /\t- /'

check-uv:
	@command -v uv >/dev/null 2>&1 || ( \
	  echo "[!] uv not found. Install with one of:"; \
	  echo "    brew install uv"; \
	  echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"; \
	  exit 1 )

install-uv: ## Install uv via Homebrew (fallback to script if brew missing)
	@if command -v uv >/dev/null 2>&1; then \
	  echo "uv already installed"; \
	elif command -v brew >/dev/null 2>&1; then \
	  brew install uv; \
	else \
	  echo "brew not found; installing via script"; \
	  curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi

py312: check-uv ## Ensure Python 3.12 is available to uv
	uv python install 3.12

setup: py312 ## Create .venv and install runtime deps
	uv sync --no-dev --python 3.12
	@echo "\n[ok] Runtime environment ready in .venv"

setup-dev: py312 ## Create .venv and install dev + runtime deps
	uv sync --python 3.12
	@echo "\n[ok] Dev environment ready in .venv"

activate: ## Print the command to activate the venv
	@echo "Run: source .venv/bin/activate"

run: ## Quick sanity check that Python runs from the env
	uv run python -V

lock: check-uv ## Resolve and write uv.lock
	uv lock

clean: ## Remove the local virtual environment
	rm -rf .venv
	@echo "Removed .venv"

mac-prereqs: ## Install macOS build prerequisites (e.g., swig for box2d)
	@if command -v brew >/dev/null 2>&1; then \
	  brew list swig >/dev/null 2>&1 || brew install swig; \
	else \
	  echo "Homebrew not found. Please install swig manually (https://brew.sh)."; \
	fi

