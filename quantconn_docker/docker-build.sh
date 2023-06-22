COMMIT=${1}
COMMIT="$(echo "${COMMIT}" | tr -d '[:space:]')"
export COMMIT
NO_TAG="ghcr.io/${2}/miccai23"
TAG="${NO_TAG}:${COMMIT}"
TAG2="${NO_TAG}:latest"
TAG="$(echo "${TAG}" | tr -d '[:space:]')"
TAG2="$(echo "${TAG2}" | tr -d '[:space:]')"

echo $TAG
docker build --no-cache -t $TAG -t $TAG2 --build-arg COMMIT ./quantconn_docker