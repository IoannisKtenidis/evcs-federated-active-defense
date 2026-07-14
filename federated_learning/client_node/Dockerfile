FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "client.py"]
