###############################################################################
# Dockerfile to build Quantconn
###############################################################################

# Use python base image
# FROM python:3.9-slim-bullseye
FROM python:3.9

RUN apt-get update \
&& apt-get install -y --no-install-recommends git \
&& apt-get purge -y --auto-remove \
&& rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED True

ARG COMMIT

# Copy and Install Quantconn
COPY ../quantconn ./
RUN pip install --no-cache-dir quantconn
# RUN pip install --no-cache-dir git+https://github.com/dipy/miccai23.git@${COMMIT}


ENTRYPOINT ["quantconn"]