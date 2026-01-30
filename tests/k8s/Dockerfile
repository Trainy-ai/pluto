# Dockerfile for k8s multinode E2E test
FROM python:3.11-slim

WORKDIR /workspace

# Install pluto and dependencies
COPY . /workspace/
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir httpx

# Copy the worker script
COPY tests/k8s/multinode_worker.py /workspace/multinode_worker.py

# Make it executable
RUN chmod +x /workspace/multinode_worker.py

# Default command
CMD ["python", "/workspace/multinode_worker.py"]
