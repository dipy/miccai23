###############################################################################
# Dockerfile to build Quantconn
###############################################################################

# Use python base image
FROM python:3.9-slim-bullseye

RUN apt-get update \
&& apt-get install -y --no-install-recommends git \
&& apt-get purge -y --auto-remove \
&& rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED True

ARG COMMIT

# Install pyAFQ
RUN pip install --no-cache-dir git+https://github.com/dipy/miccai23.git@${COMMIT}

ENTRYPOINT ["quantconn"]