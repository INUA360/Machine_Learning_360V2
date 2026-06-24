FROM python:3.12-slim

WORKDIR /app

# Ensure output is sent straight to terminal without buffering
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8001

CMD ["uvicorn", "machine_learning_360v2.app:app", "--host", "0.0.0.0", "--port", "8001"]
