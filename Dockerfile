FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# libgomp1 is the OpenMP runtime that XGBoost needs on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Serving dependencies only (no MLflow, pytest or plotting libraries).
COPY requirements-serve.txt .
RUN pip install --no-cache-dir --default-timeout=120 --retries 5 -r requirements-serve.txt

COPY app.py loan_features.py ./
COPY model/ ./model/
COPY static/ ./static/

# Run as a non-root user; logs/ must be writable for prediction logging.
RUN useradd --create-home appuser && mkdir -p logs && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"8000\")}/health')"

# sh -c expands $PORT (Render/Railway inject it; default 8000). exec makes uvicorn PID 1 so it receives stop signals.
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT}"]