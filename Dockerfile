FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY velmora_env/ ./velmora_env/
COPY server/ ./server/
COPY inference.py .
COPY app.py .
COPY openenv.yaml .
COPY pyproject.toml .

RUN pip install --no-cache-dir -e .

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
