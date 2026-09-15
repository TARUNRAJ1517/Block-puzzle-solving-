FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
EXPOSE 10000
CMD ["streamlit","run","app/main.py","--server.address=0.0.0.0","--server.port=10000","--server.headless=true"]
