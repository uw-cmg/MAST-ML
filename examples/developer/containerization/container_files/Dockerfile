# Initial setup
FROM ubuntu

# Make any installation noninteractive just in case
ENV DEBIAN_FRONTEND=noninteractive

# Update and upgrade packages
RUN apt update
RUN apt upgrade -y

# Install needed software
RUN apt install python3-pip git -y
RUN pip install -U pip
RUN pip install git+https://github.com/uw-cmg/MAST-ML.git@dev_lane

# Define the working directory
WORKDIR /home

# Copy important files to build container
COPY X_train.csv .
COPY y_train.csv .
COPY predict.py .
COPY RandomForestRegressor.pkl .
COPY domain_madml.pkl .
COPY recalibration_parameters_train.csv .
COPY StandardScaler.pkl .
