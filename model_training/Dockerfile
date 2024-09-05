# Use the official TensorFlow GPU image as a base image
FROM tensorflow/tensorflow:2.14.0-gpu

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Set the environment variable to allow TensorFlow to use all available GPUs
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# Run the script when the container launches
CMD ["python", "./model_training.py"]