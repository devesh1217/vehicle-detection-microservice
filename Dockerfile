# Start from the official Python image
FROM python:3.10-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Set the working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy YOLO model files and the app folder
COPY app/ app/

# Expose the port the app runs on
EXPOSE 5000

# Run the Flask application
CMD ["python", "app/main.py"]


# Command to run this microservice
# Build the Docker image
# docker build -t my-flask-app .

# Run the Docker container
# docker run -p 5000:5000 my-flask-app