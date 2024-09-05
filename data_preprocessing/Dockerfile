# Use the Python image (for now)
FROM python:3.12.4

# Install Java (OpenJDK) - required for PySpark
RUN apt-get update && \
    apt-get install -y openjdk-17-jdk && \
    apt-get clean;

# Set environment variables for Java
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64
ENV PATH=$JAVA_HOME/bin:$PATH

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy PySpark script into the working directory
COPY data_preprocessing.py .

# Copy CSV file (temporary) 
COPY Friday-WorkingHours-Morning.pcap_ISCX.csv .

# Set the default command to run the PySpark job with Ivy home configuration
ENTRYPOINT ["python", "/app/data_preprocessing.py", "1"]