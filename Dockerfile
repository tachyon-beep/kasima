FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir clearml
COPY . .
ENV CLEARML__API__API_SERVER=${CLEARML__API__API_SERVER}
ENV CLEARML__API__WEB_SERVER=${CLEARML__API__WEB_SERVER}
ENV CLEARML__API__FILES_SERVER=${CLEARML__API__FILES_SERVER}
CMD ["python", "-m", "scripts.run_experiment", "--device", "cpu", "--amp"]
