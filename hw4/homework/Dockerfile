FROM svizor/zoomcamp-model:mlops-3.10.0-slim

WORKDIR /app
COPY [ "model2.bin", "model.bin" ]

COPY ['Pipfile', 'Pipfile.lock', "./"]

RUN pipenv install --system --deploy

COPY ['starter.py', "./"]

RUN python3 starter.py 2022 04