FROM python:3.10-slim

WORKDIR /app

# Install system deps — libpq-dev is needed for psycopg2 (PostgreSQL client)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose both API (8000) and Streamlit (8501) ports
EXPOSE 8000 8501

# Default: run the FastAPI backend, reading from the PORT environment variable if set
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]