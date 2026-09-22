FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --default-timeout=100 -e ".[dev]"


COPY . .

CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
