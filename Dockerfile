# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (optional but useful for puLP solver)
RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get clean

# Copy only requirements.txt first for caching
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy whole app
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]