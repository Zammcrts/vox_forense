FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends default-jre-headless ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
# /tmp es tmpfs (ver docker-compose.yml): el archivo temporal que owlready2 le pasa a HermiT vive solo en RAM
CMD ["uvicorn", "vox.api:app", "--host", "0.0.0.0", "--port", "8000"]
