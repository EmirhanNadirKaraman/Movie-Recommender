FROM python:3.9-alpine

RUN apk add --no-cache --update \
    cmake \
    build-base linux-headers \
    python3 python3-dev gcc \
    gfortran musl-dev \
    libffi-dev openssl-dev

# Create app directory
WORKDIR /app

RUN pip install --upgrade --pre pip

# Install app dependencies
COPY requirements.txt ./

RUN pip install -r requirements.txt

# Bundle app source
COPY . .

EXPOSE 5000
CMD [ "python3", "main.py"]