FROM python:3.13-slim-bookworm

ARG BUILD_VERSION=0.2.1
ARG BUILD_ARCH=amd64
# This is the unchanged upstream source reviewed for this App release.
ARG UPSTREAM_REVISION=9c0c17c6eb6af297b1ba933af930c0acf3689ff9
LABEL io.hass.version="${BUILD_VERSION}" io.hass.type="addon" io.hass.arch="${BUILD_ARCH}"

ENV PYTHONUNBUFFERED=1 EPEXPREDICTOR_DATADIR=/data
WORKDIR /code
ADD https://github.com/b3nn0/EpexPredictor/archive/${UPSTREAM_REVISION}.tar.gz /tmp/upstream.tar.gz
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && tar -xzf /tmp/upstream.tar.gz --strip-components=1 -C /code \
    && rm /tmp/upstream.tar.gz \
    && pip install --no-cache-dir -r requirements.txt

COPY run.py /run.py
COPY ha_app.py /code/ha_app.py
# No Supervisor watchdog: upstream completes initial data loading/training before
# accepting HTTP requests. An eager watchdog would repeatedly interrupt that work.
HEALTHCHECK --interval=60s --timeout=10s --start-period=60m --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/openapi.json', timeout=5).close()"
CMD ["python", "/run.py"]
