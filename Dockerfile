# Stage 1: Builder
FROM python:3.11.9-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.11.9-slim

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy installed dependencies from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser api.py config.py ./

# Make sure scripts in .local are usable
ENV PATH=/home/appuser/.local/bin:$PATH

USER appuser

EXPOSE 8000

# The model isn't trained at build time anymore; it will be supplied via volume or mounted config
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
