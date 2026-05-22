# Vietnamese Graph RAG — serving image (CPU). GPU: dùng base nvidia/cuda + torch CUDA.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# system deps cho underthesea / build
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
RUN pip install --upgrade pip && pip install .

COPY app ./app
COPY config.yaml ./

# artifacts (index + KG) nên được build sẵn rồi mount vào, hoặc build lúc khởi động
EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
