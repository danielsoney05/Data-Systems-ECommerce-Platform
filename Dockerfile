# Use official Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy files
COPY requirements.txt .
COPY input_app/key.json /app/input_app/key.json
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose Flask port
EXPOSE 5000

# Environment variables
ENV PYTHONUNBUFFERED=1

# Run app
CMD ["python", "input_app/app.py"]