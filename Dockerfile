FROM python:3.10-slim

WORKDIR /app

# System deps for spaCy and PyTorch
RUN apt-get update && apt-get install -y build-essential wget curl git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "entity_aware_fnd.cli.infer", "--text", "Sample"]
