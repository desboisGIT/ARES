FROM python:3.12-slim

# Optional CPU-only learning stack; default image keeps the Phase 3 flow.
ARG ARES_FLIGHT=0

# The verified CPU worker compiles a small read-only arithmetic-mode probe.
RUN if [ "$ARES_FLIGHT" = "1" ]; then \
      apt-get update \
      && apt-get install -y --no-install-recommends gcc libc6-dev \
      && rm -rf /var/lib/apt/lists/*; \
    fi

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
COPY . .
RUN python -m pip install --no-cache-dir 'setuptools>=68' wheel \
    && python -m pip install --no-cache-dir '.[simulation]' \
    && if [ "$ARES_FLIGHT" = "1" ]; then \
         python -m pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
         && python -m pip install --no-cache-dir '.[flight]'; \
       fi \
    && python -m unittest discover -s tests -v \
    && python -m ares validate \
    && python -m ares phase0 --output reports/phase_0.json \
    && python -m ares benchmark --check \
    && python -m ares phase1 --count 30 --workers 1 --smoke --output reports/phase_1_smoke.json \
    && python -m ares phase2 --backend cpu --batch-size 4 --frames 12 --output reports/phase_2_smoke.json \
    && python -m ares phase3 --backend cpu --num-envs 4 --benchmark-steps 5 --episodes 1 --start-seed 100 --split train --smoke --output reports/phase_3_smoke.json \
    && if [ "$ARES_FLIGHT" = "1" ]; then \
         python -m ares control-check --device cpu --count 8 --duration 4 --settle 2 --output reports/control_smoke.json \
         && python -m ares phase4 --device cpu --train-updates 1 --batch-size 2 --rollout-steps 4 --episodes 2 --max-steps 10 --smoke --output reports/phase_4_repair_smoke.json \
         && python -m ares.verified_safety smoke --output reports/phase_4_verified_safety_smoke.json; \
       fi \
    && python -m pip install --no-deps --no-build-isolation . \
    && groupadd --gid 10001 ares \
    && useradd --uid 10001 --gid ares --create-home ares \
    && mkdir -p /app/reports \
    && chown -R ares:ares /app/reports
USER ares
CMD ["python", "-m", "ares", "phase3", "--backend", "cpu", "--num-envs", "4", "--benchmark-steps", "5", "--episodes", "1", "--start-seed", "100", "--split", "train", "--smoke", "--output", "reports/phase_3_runtime.json"]
