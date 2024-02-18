FROM python:3.11-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app
COPY ./controller /code/controller
COPY ./service /code/service
COPY ./model /code/model
COPY ./filter_model /code/filter_model

EXPOSE 10101

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10101"]