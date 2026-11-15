FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY src ./src
ARG INSTALL_ML=false
RUN if [ "$INSTALL_ML" = "true" ]; then pip install '.[ml,diarization]'; else pip install .; fi
COPY app.py ./
COPY .streamlit ./.streamlit
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD curl --fail http://localhost:8501/_stcore/health || exit 1
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
