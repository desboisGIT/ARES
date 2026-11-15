PYTHON ?= .venv/bin/python
CONTAINER_ENGINE ?= docker
WORLD_WORKERS ?= 1

.PHONY: test check build container phase1 phase2 phase3 phase4 phase4-validation phase4-smoke phase4-container control-check control-validation

test:
	$(PYTHON) -m unittest discover -s tests -v

check: test
	$(PYTHON) -m ares validate
	$(PYTHON) -m ares phase0 --output reports/phase_0.json
	$(PYTHON) -m ares benchmark --check
	$(PYTHON) -m ares phase1 --count 30 --workers 1 --smoke --output reports/phase_1_smoke.json
	$(PYTHON) -m ares phase2 --backend cpu --batch-size 4 --frames 12 --output reports/phase_2_smoke.json
	$(PYTHON) -m ares phase3 --backend cpu --num-envs 4 --benchmark-steps 5 --episodes 1 --start-seed 100 --split train --smoke --output reports/phase_3_smoke.json

build: check
	$(PYTHON) -m pip wheel --no-deps --wheel-dir dist .

container:
	$(CONTAINER_ENGINE) build -t ares:phase3 .

phase4-container:
	$(CONTAINER_ENGINE) build --build-arg ARES_FLIGHT=1 -t ares:phase4 .

phase1:
	$(PYTHON) -m ares phase1 --count 10000 --workers $(WORLD_WORKERS) --output reports/phase_1.json

# The CUDA target requires its optional extra in the selected environment.
phase2:
	$(PYTHON) -m ares phase2 --backend cuda --batch-size 4096 --frames 12 --output reports/phase_2.json

phase3:
	$(PYTHON) -m ares phase3 --backend cuda --num-envs 4096 --benchmark-steps 25 --episodes 100 --start-seed 50000 --split test --workers 4 --output reports/phase_3.json

# Checks the learned waypoint gate; shield and hardware gates remain separate.
# Seed 80000 is sealed evidence: use a new cohort after any further tuning.
phase4:
	$(PYTHON) -m ares phase4 --device cuda --algorithms bptt ppo --residual-acceleration .5 --residual-penalty 1 --guidance-speed-fraction .95 --train-updates 50 --batch-size 64 --rollout-steps 128 --episodes 1000 --max-steps 600 --eval-start-seed 80000 --output reports/phase_4_gate_runtime.json

phase4-validation:
	$(PYTHON) -m ares phase4 --device cuda --algorithms bptt ppo --residual-acceleration .5 --residual-penalty 1 --guidance-speed-fraction .95 --train-updates 50 --batch-size 64 --rollout-steps 128 --episodes 256 --max-steps 600 --eval-split validation --eval-start-seed 46000 --smoke --output reports/phase_4_gate_development.json

phase4-smoke:
	$(PYTHON) -m ares phase4 --device cpu --train-updates 1 --batch-size 2 --rollout-steps 4 --episodes 2 --max-steps 10 --smoke --output reports/phase_4_repair_smoke.json

# These targets require the optional flight extra in the selected environment.
control-check:
	$(PYTHON) -m ares control-check --device cpu --count 8 --duration 4 --settle 2 --output reports/control_smoke.json

control-validation:
	$(PYTHON) -m ares control-check --device cuda --count 512 --duration 4 --settle 2 --batch-size 64 --compile-dynamics --output reports/control_validation.json

# Bootstrap only the default interpreter. Explicit PYTHON overrides are managed
# by the caller (CI installs .[simulation] before invoking make).
ifeq ($(PYTHON),.venv/bin/python)
test check: .venv/.simulation-ready

.venv/.simulation-ready: pyproject.toml
	test -x .venv/bin/python || python3 -m venv .venv
	.venv/bin/python -m pip install -e '.[simulation]'
	touch $@
endif
.PHONY: verified-safety-smoke verified-safety-validation

# Conditional backup component only; the integrated Phase 4 gate is separate.
verified-safety-smoke:
	$(PYTHON) -m ares.verified_safety smoke $(if $(SHIELD_OUTPUT),--output "$(SHIELD_OUTPUT)")

verified-safety-validation:
	$(PYTHON) -m ares.verified_safety validate $(if $(SHIELD_OUTPUT),--output "$(SHIELD_OUTPUT)")
