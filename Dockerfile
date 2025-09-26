# Dockerfile (sin extensión)
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

# Paquetes base
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar deps primero (mejora cache)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copiar el proyecto
COPY . /app

EXPOSE 8000

# Arranque de la API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
