FROM python:3.9-slim

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y libatlas-base-dev

RUN pip install --no-cache-dir tflite-runtime==2.12.0

RUN pip install -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
