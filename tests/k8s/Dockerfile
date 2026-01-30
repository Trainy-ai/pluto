# Dockerfile for k8s multinode E2E test
# Based on konduktor CI pattern
FROM python:3.11-slim

WORKDIR /workspace

# Install PyTorch (CPU) for torchrun and pluto
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install pluto and dependencies
COPY . /workspace/
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir httpx

# Copy the worker script
COPY tests/k8s/multinode_worker.py /workspace/multinode_worker.py

# Make it executable
RUN chmod +x /workspace/multinode_worker.py

# Default command (will be overridden by konduktor run command)
CMD ["python", "/workspace/multinode_worker.py"]
