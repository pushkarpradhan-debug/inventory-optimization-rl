# server/Dockerfile
# Builds a Docker container that runs the FastAPI environment server.
# Build: docker build -t inventory-restock-env -f server/Dockerfile .
# Run:   docker run -p 7860:7860 inventory-restock-env

FROM python:3.10

# Create non-root user
RUN useradd -m -u 1000 user
USER user

# Set PATH
ENV PATH="/home/user/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy files
COPY --chown=user . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]