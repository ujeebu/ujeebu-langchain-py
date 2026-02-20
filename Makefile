.PHONY: all install install-dev install-test test test-unit test-integration lint format type-check clean build publish-test publish help

# Default target
all: help

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-test:
	pip install -e ".[test]"

install-all:
	pip install -e ".[dev,test,test_integration]"

# Testing targets
test: test-unit

test-unit:
	pytest tests/unit_tests/ -v

test-integration:
	pytest tests/integration_tests/ -v

test-all:
	pytest tests/ -v

test-coverage:
	pytest tests/ -v --cov=langchain_ujeebu --cov-report=html --cov-report=term

# Code quality targets
lint:
	flake8 langchain_ujeebu tests

format:
	black langchain_ujeebu tests examples

format-check:
	black langchain_ujeebu tests examples --check

type-check:
	mypy langchain_ujeebu

# Cleaning targets
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -delete

# Build and publish targets
build: clean
	python -m build

publish-test: build
	python -m twine upload --repository testpypi dist/*

publish: build
	python -m twine upload dist/*

# Help target
help:
	@echo "LangChain Ujeebu Integration - Available targets:"
	@echo ""
	@echo "Installation:"
	@echo "  install          Install package in development mode"
	@echo "  install-dev      Install with development dependencies"
	@echo "  install-test     Install with test dependencies"
	@echo "  install-all      Install with all dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test             Run unit tests (default)"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests (requires UJEEBU_API_KEY)"
	@echo "  test-all         Run all tests"
	@echo "  test-coverage    Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint             Run flake8 linter"
	@echo "  format           Format code with black"
	@echo "  format-check     Check code formatting"
	@echo "  type-check       Run mypy type checker"
	@echo ""
	@echo "Build & Publish:"
	@echo "  clean            Remove build artifacts"
	@echo "  build            Build distribution packages"
	@echo "  publish-test     Publish to TestPyPI"
	@echo "  publish          Publish to PyPI"

