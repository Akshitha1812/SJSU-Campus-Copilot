FROM python:3.10-slim-buster

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# (1) System deps so pip can compile wheels if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc python3-dev \
    && rm -rf /var/lib/apt/lists/*

# (2) Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# (3) Copy the rest of your code
COPY . .

# (4) Start the app (WORKDIR=/app)
CMD ["python3", "backend/app.py"]



