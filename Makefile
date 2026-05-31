PYTHON ?= python3

ifneq ($(wildcard .venv/bin/python),)
FIGURE_PYTHON ?= .venv/bin/python
else
FIGURE_PYTHON ?= python3
endif

ifneq ($(wildcard /opt/homebrew/Cellar/expat/2.8.0/lib/libexpat.1.dylib),)
FIGURE_ENV ?= MPLCONFIGDIR=$(CURDIR)/.matplotlib-cache DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/expat/2.8.0/lib
else
FIGURE_ENV ?= MPLCONFIGDIR=$(CURDIR)/.matplotlib-cache
endif

.PHONY: figures validate tables tests reproduce assets clean-cache

figures:
	$(FIGURE_ENV) $(FIGURE_PYTHON) scripts/cascade_figures.py
	$(FIGURE_ENV) $(FIGURE_PYTHON) scripts/finite_squeezing_span_limit.py
	$(FIGURE_ENV) $(FIGURE_PYTHON) scripts/ordering_effect.py
	$(FIGURE_ENV) $(FIGURE_PYTHON) scripts/stochastic_outage.py
	$(FIGURE_ENV) $(FIGURE_PYTHON) scripts/asymmetric_cascade.py
	$(FIGURE_ENV) $(FIGURE_PYTHON) scripts/operational_rates.py

validate:
	$(PYTHON) scripts/validate_cascade.py
	$(PYTHON) scripts/validate_asymmetric_formula.py

tables:
	$(PYTHON) scripts/n_max_table.py
	$(PYTHON) scripts/optimal_ordering.py

tests:
	$(PYTHON) -m pytest tests

assets:
	$(PYTHON) scripts/manuscript_assets.py

reproduce: figures validate tables tests assets

clean-cache:
	rm -rf .matplotlib-cache .pytest_cache
