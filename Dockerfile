###############################################################################
# Dockerfile to build Quantconn
###############################################################################

# Use python base image
# FROM python:3.9-slim-bullseye
FROM python:3.9

# RUN apt-get update \
# && apt-get install -y --no-install-recommends git \
# && apt-get purge -y --auto-remove \
# && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED True

# Set the working directory in the container
WORKDIR /quantconn
COPY . .

# ARG COMMIT

RUN pip install --upgrade pip
RUN pip install --no-cache-dir packaging numpy scipy nibabel h5py tqdm
RUN pip install --no-cache-dir dipy
RUN pip install --no-cache-dir .
# RUN --mount=source=.git,target=.git,type=bind \
#     pip install --no-cache-dir .
# RUN pip install --no-cache-dir git+https://github.com/dipy/miccai23.git@${COMMIT}


ENTRYPOINT ["quantconn"]