FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Copy source
COPY src/ ./src/

# Install package
RUN pip install -e .

CMD ["python", "-c", "import deepiri_modelkit; print('ModelKit ready')"]

