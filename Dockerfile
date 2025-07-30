FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy app code
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default command (override for different CLI args)
CMD ["python", "cli.py"]
