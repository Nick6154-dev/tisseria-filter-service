FROM python:3.11

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY ./app /code/app
COPY ./controller /code/controller
COPY ./service /code/service
COPY ./model /code/model
COPY ./filter_model /code/filter_model

CMD ["uvicorn", "app.main:app", "--port", "10101"]