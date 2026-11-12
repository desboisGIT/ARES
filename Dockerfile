FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
COPY . .
RUN python -m pip install --no-cache-dir 'setuptools>=68' wheel \
    && python -m unittest discover -s tests -v \
    && python -m ares validate \
    && python -m ares phase0 --output reports/phase_0.json \
    && python -m ares benchmark --check \
    && python -m pip install --no-deps --no-build-isolation . \
    && groupadd --gid 10001 ares \
    && useradd --uid 10001 --gid ares --create-home ares \
    && mkdir -p /app/reports \
    && chown -R ares:ares /app/reports
USER ares
CMD ["python", "-m", "ares", "phase0", "--output", "reports/phase_0.json"]
