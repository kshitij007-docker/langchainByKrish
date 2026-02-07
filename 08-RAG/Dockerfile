FROM python:3.11-slim

# Make Python + pip quieter and avoid caching
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

# Install minimal build tools (needed for some Python deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code
COPY . .

# Streamlit default port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "logsReader.py", "--server.port=8501", "--server.address=0.0.0.0"]
