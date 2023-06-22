###############################################################################
# Dockerfile to build Quantconn
###############################################################################

# Use python base image
FROM python:3.9-slim-bullseye

ENV PYTHONUNBUFFERED True

ARG COMMIT

# Install pyAFQ
RUN pip install --no-cache-dir git+https://github.com/dipy/miccai23.git@${COMMIT}

ENTRYPOINT ["quantconn"]