FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1

COPY flask_app/ /app/

RUN pip install -r docker_requirements.txt

RUN python -m nltk.downloader stopwords wordnet

EXPOSE 8000

CMD ["python", "app.py"]