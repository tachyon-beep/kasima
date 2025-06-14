FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
ENV CLEARML__API__API_SERVER=http://localhost:8008 \
    CLEARML__API__WEB_SERVER=http://localhost:8080 \
    CLEARML__API__FILES_SERVER=http://localhost:8081
COPY . .
CMD ["python", "-m", "scripts.run_experiment", "--device", "cpu", "--amp"]
