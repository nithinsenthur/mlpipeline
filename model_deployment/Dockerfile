# Use TensorFlow image as a base image
FROM python:3.12.4

# Install HDF5 library and development headers
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    python3-dev \
    build-essential

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run the script when the container launches
CMD ["python", "./model_deployment.py"]