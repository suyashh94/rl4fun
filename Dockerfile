FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/root/.local/bin:${PATH}"

# System packages and Python 3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git pkg-config build-essential swig \
    python3.12 python3.12-venv python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package/dependency manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

# Copy dependency manifests first for better layer caching
COPY pyproject.toml* uv.lock* ./

# Create a project-local venv and install only runtime deps with Python 3.12
RUN uv sync --no-dev --no-install-project --python 3.12

# Make the venv active for subsequent layers and runtime
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:${PATH}"

# Copy the rest of the project
COPY . .

# Default command (override as needed)
CMD ["python", "--version"]
