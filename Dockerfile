FROM python:3.11-slim

WORKDIR /app


# Bedd runtime (Bun-style) — glibc binary
ARG BEDD_IMAGE=ghcr.io/team-deepiri/bedd:0.6
COPY --from= /usr/local/bin/bedd /usr/local/bin/bedd
COPY --from= /opt/bedd/skills /opt/bedd/skills
ENV BEDD_SKILLS_DIR=/opt/bedd/skills


# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Copy source
COPY src/ ./src/

# Install package
RUN pip install -e .

CMD ["python", "-c", "import deepiri_modelkit; print('ModelKit ready')"]

