FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install  
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy all project files
COPY . /app/

# Set PYTHONPATH
ENV PYTHONPATH="/app:$PYTHONPATH"

# Enable web interface
ENV ENABLE_WEB_INTERFACE=true

# HuggingFace Spaces expects port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# Run the server on port 7860 (HF Spaces requirement)
CMD ["uvicorn", "cyber_range.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
