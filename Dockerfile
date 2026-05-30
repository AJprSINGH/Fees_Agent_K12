FROM python:3.11-slim

WORKDIR /app

# Copy all project files
COPY . .

# Install system dependencies for openpyxl + lightgbm
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Hugging Face Spaces expects port 7860
EXPOSE 7860

# Use app.py — startup auto-trains models if pkl files are missing
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--timeout-keep-alive", "120"]
