# Happiness in a container: the web chat on port 8000, your data in /data.
#
#   docker build -t happiness .
#   docker run -d --name happiness -p 8000:8000 \
#     -e ANTHROPIC_API_KEY=sk-ant-... -e HAPPINESS_PASSWORD='a long passphrase' \
#     -v happiness-data:/data happiness

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HAPPINESS_HOME=/data

RUN useradd --create-home --uid 1000 happiness \
 && mkdir /data \
 && chown happiness:happiness /data

WORKDIR /app
COPY requirements.txt /app/happiness/requirements.txt
RUN pip install --no-cache-dir -r /app/happiness/requirements.txt
COPY . /app/happiness

USER happiness
VOLUME /data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)"]
CMD ["python", "-m", "happiness", "serve", "--host", "0.0.0.0", "--port", "8000"]
