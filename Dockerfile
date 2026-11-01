# syntax=docker/dockerfile:1.7
# One image, many roles: API, worker, MLflow server, bootstrap job and traffic generator
# all run from this image with a different command (see docker-compose.yml).

# ---- build stage: resolve + compile dependencies into a venv --------------------------------
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---- runtime stage: no compilers, no build cache, non-root ---------------------------------
FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    MLOPS_CONFIG=/app/configs/config.yaml \
    GIT_PYTHON_REFRESH=quiet

RUN useradd --create-home --uid 10001 mlops \
 && mkdir -p /mlflow/artifacts \
 && chown -R mlops:mlops /mlflow
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY configs ./configs
COPY src ./src

# Named volumes mounted here inherit this ownership on first use.
VOLUME ["/mlflow/artifacts"]
USER mlops
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=40s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=2).status == 200 else 1)"]

# One uvicorn worker per container: scale with replicas (keeps the Prometheus counters simple).
CMD ["uvicorn", "mlops.serving.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
