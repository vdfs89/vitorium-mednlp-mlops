# VITORIUM — imagem da API de inferencia (multi-stage).
#
#   Estagio 1 (builder): instala as dependencias de runtime num virtualenv
#                        isolado em /opt/venv. Todas as dependencias tem wheels
#                        manylinux, entao nao ha compiladores nesta imagem.
#   Estagio 2 (runtime): parte de um python:3.11-slim limpo e copia apenas o
#                        venv + src/ + models/. Nada de pip cache, requirements
#                        ou arquivos de build vai para a imagem final.
#
# Build:  docker build -t vitorium-api:latest .            (contexto = raiz)
# Compose: docker compose -f docker/docker-compose.yml up  (context: ..)
#
# O backend de inferencia (VITORIUM_BACKEND=sklearn|onnx) NAO e fixado aqui:
# e definido em tempo de execucao pelo docker-compose.yml / `docker run -e`.

# ---------------------------------------------------------------------------
# Estagio 1 — builder
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Estagio 2 — runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.source="https://github.com/vdfs89/vitorium-mednlp-mlops" \
      org.opencontainers.image.title="VITORIUM API" \
      org.opencontainers.image.description="API FastAPI de triagem de textos medicos (TF-IDF + LogisticRegression, backends sklearn/ONNX) com metricas Prometheus"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}"

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser models/ models/

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
