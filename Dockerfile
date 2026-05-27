# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS builder

WORKDIR /build

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip build && \
    python -m build --wheel

FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="modelmeld"
LABEL org.opencontainers.image.description="ModelMeld open-source AI gateway"
LABEL org.opencontainers.image.licenses="AGPL-3.0-or-later"
LABEL org.opencontainers.image.source="https://github.com/ModelMeld/modelmeld"

RUN useradd -r -u 1000 -m gateway

WORKDIR /app
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

USER gateway
EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=6 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8080/healthz').status==200 else 1)" || exit 1

CMD ["python", "-m", "modelmeld"]
