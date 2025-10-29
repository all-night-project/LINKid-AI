FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY data ./data
COPY src ./src
COPY test_implementation.py ./
COPY tests ./tests

CMD ["python", "-u", "test_implementation.py"]
