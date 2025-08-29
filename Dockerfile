FROM python:3.10-slim-buster

WORKDIR /app/backend

COPY . /app

RUN pip install -r requirements.txt

CMD ["python3","/app.py"]

