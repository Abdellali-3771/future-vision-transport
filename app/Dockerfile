FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 && rm -rf /var/lib/apt/lists/*

# Dossier de travail dans le conteneur
WORKDIR /app

# Copie du fichier requirements et installation
COPY app/requirements.txt .
RUN pip install -r requirements.txt

# Copie de tout le code et des poids
COPY app /app

EXPOSE 8000
CMD ["python", "-m", "backend.main"]
