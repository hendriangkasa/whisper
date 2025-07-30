# Use a standard RunPod base image that already has PyTorch and CUDA
FROM runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel

# Set a working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install our Python dependencies, including whisper-timestamped
RUN pip install --no-cache-dir -r requirements.txt

# Copy our handler code into the container
COPY handler.py .

# RunPod's base image knows how to start the handler.py automatically.
