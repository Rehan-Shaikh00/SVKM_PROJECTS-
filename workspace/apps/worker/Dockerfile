FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --default-timeout=100 -e ".[dev]"


COPY . .

CMD ["python", "-m", "apps.worker.main"]
