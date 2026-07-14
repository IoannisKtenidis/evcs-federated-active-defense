# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed for some Python packages (like h5py, etc.)
RUN apt-get update --fix-missing && apt-get install -y \
    build-essential \
    python3-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Create directories for uploads and results
RUN mkdir -p uploads results

# Expose port 5000 for the Flask API
EXPOSE 5000

# Run the API server
CMD ["python", "white_box/api_server.py"]
