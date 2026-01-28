# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml uv.lock* ./
COPY app/ ./app/
COPY data/ ./data/

# Install dependencies using uv
RUN uv sync --no-dev

# Copy environment file
COPY .env* ./

# Set environment variable to ensure Python output is sent straight to logs
ENV PYTHONUNBUFFERED=1

# Activate virtual environment and run the app
CMD [".venv/bin/python", "-m", "app"]
