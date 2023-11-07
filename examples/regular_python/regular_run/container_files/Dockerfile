# Initial setup
FROM ubuntu

# Make any installation noninteractive just in case
ENV DEBIAN_FRONTEND=noninteractive

# Update and upgrade packages
RUN apt update
RUN apt upgrade -y

# Install and update python3 and corresponding pip
RUN apt install python3-pip -y
RUN pip install -U pip

# Define the working directory
WORKDIR /home

# Copy important files to build container
COPY X_train.csv .
COPY predict.py .
COPY RandomForestRegressor.pkl .
COPY domain_madml.pkl .
COPY recalibration_parameters_train.csv .
COPY StandardScaler.pkl .

# Install other python packages user should start from clean environment
# and run "pip freeze > requirements.txt"
COPY requirements.txt .
RUN pip install -r requirements.txt
