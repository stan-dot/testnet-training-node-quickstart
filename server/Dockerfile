# restapi/Dockerfile
FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04 as server
# FROM python:3.11-slim


RUN apt-get update && apt-get install -y \
    python3-dev \
    python3-pip \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first to leverage Docker cache
# COPY requirements.txt .
# RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
# COPY . .

CMD ["python", "hello.py"]
# CMD ["python", "full_automation.py"]

# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
# FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04

# RUN apt-get update && apt-get install -y \
#     python3-dev \
#     python3-pip \
#     && ln -s /usr/bin/python3 /usr/bin/python \
#     && rm -rf /var/lib/apt/lists/*

# WORKDIR /app

# # Copy only requirements first to leverage Docker cache
# COPY requirements.txt .
# RUN pip3 install --no-cache-dir -r requirements.txt

# # Copy the rest of the application code
# COPY . .

# CMD ["python", "full_automation.py"]