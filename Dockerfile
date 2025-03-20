FROM python:3.9-slim

WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models/model.pkl
ENV MODEL_VERSION=latest

# Expose the port for the API
EXPOSE 8000

# Run the model service when the container launches
CMD ["python", "-m", "src.models.predict_model", "--serve"]
