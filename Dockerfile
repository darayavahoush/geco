# Multi-stage build: compile the React frontend, then serve it + the API from
# one FastAPI container. Hugging Face Spaces (Docker SDK) expects the app to
# listen on port 7860.

FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY src ./src
COPY backend ./backend
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# HF Spaces caches under /data if persistent storage is enabled; otherwise
# torch.hub re-downloads DINOv2 weights (~330MB) on every cold start.
ENV TORCH_HOME=/app/.cache/torch

EXPOSE 7860
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
