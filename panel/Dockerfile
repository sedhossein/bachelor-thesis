

FROM python:3.8.3-alpine

ENV MICRO_SERVICE=/home/app/dashboard

# set work directory
RUN mkdir -p $MICRO_SERVICE
RUN mkdir -p $MICRO_SERVICE/staticfiles

# where the code lives
WORKDIR $MICRO_SERVICE

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install psycopg2 dependencies
RUN apk update \
    && apk add --virtual build-deps gcc python3-dev musl-dev \
    && apk add postgresql-dev gcc python3-dev musl-dev \
    && apk del build-deps \
    && apk --no-cache add musl-dev linux-headers g++

# install dependencies
RUN pip install --upgrade pip

# copy project
COPY . $MICRO_SERVICE
RUN pip install -r requirements.txt
COPY ./entrypoint.sh $MICRO_SERVICE

CMD ["/bin/bash", "/home/app/dashboard/entrypoint.sh"]




