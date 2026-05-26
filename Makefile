.PHONY: visual-tests

PYTHON ?= python
REPO_ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
export PYTHONPATH := $(REPO_ROOT)

visual-tests:
	$(PYTHON) visual_tests/01_endpoint_health.py
	$(PYTHON) visual_tests/02_pbp_schema.py
