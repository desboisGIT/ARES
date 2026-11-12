PYTHON ?= python3
CONTAINER_ENGINE ?= docker

.PHONY: test check build container

test:
	$(PYTHON) -m unittest discover -s tests -v

check: test
	$(PYTHON) -m ares validate
	$(PYTHON) -m ares phase0 --output reports/phase_0.json
	$(PYTHON) -m ares benchmark --check

build: check
	$(PYTHON) -m pip wheel --no-deps --wheel-dir dist .

container:
	$(CONTAINER_ENGINE) build -t ares:phase0 .
