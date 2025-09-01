# Base Python
FROM python:3.9-slim

# Exponer puerto
EXPOSE 5002

# Directorio de trabajo
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# instalar pytest para pruebas
RUN pip install --no-cache-dir pytest pytest-flask

COPY app.py .
COPY ml/ ./ml
COPY templates/ ./templates

# Instalar dependencias del sistema y certificados SSL
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsm6 libxext6 ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

# Actualizar pip, setuptools y wheel
RUN python -m pip install --upgrade pip setuptools wheel

# Instalar paquetes Python
RUN pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

# Comando para iniciar Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5002", "app:app"]
