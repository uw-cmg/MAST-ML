# Initial setup
FROM ubuntu

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt upgrade -y

RUN apt install python3-pip -y
RUN pip install -U pip

WORKDIR /home

COPY X_train.csv .
COPY predict.py .
COPY model.pkl .
COPY calibration_file.csv .
COPY preprocessor.pkl .
COPY requirements.txt .

RUN pip install -r requirements.txt
