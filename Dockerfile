FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install supervisor for process management
RUN apt-get update \ 
    && apt-get install -y supervisor \ 
    && rm -rf /var/lib/apt/lists/*

# Copy app code
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Add Supervisor configuration
COPY supervisord.conf /etc/supervisor/conf.d/trading_app.conf

# Run Supervisor in foreground so container stays alive
CMD ["supervisord", "-n"]
