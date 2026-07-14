FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["python", "server.py"]
