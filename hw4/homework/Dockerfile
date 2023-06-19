FROM svizor/zoomcamp-model:mlops-3.10.0-slim

RUN pip install -U pip
RUN pip install pipenv
RUN pip install pandas
RUN pip install pyarrow

WORKDIR /app

RUN mkdir outputs

COPY ["Pipfile", "./"]
COPY ["Pipfile.lock", "./"]
RUN pipenv install --system --deploy

COPY ["starter.py", "./"]

RUN python3 starter.py 2022 04