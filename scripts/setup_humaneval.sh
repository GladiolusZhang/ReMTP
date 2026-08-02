#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${HUMANEVAL_DOCKER_IMAGE:-python:3-slim}"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required for isolated HumanEval execution." >&2
  exit 2
fi
if ! docker info >/dev/null 2>&1; then
  echo "Docker is installed but the daemon is unavailable." >&2
  exit 2
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  if [[ "${PULL_IMAGE:-1}" != "1" ]]; then
    echo "Missing Docker image: $IMAGE" >&2
    echo "Run with PULL_IMAGE=1 or pull it manually." >&2
    exit 2
  fi
  docker pull "$IMAGE"
fi

"$PROJECT_DIR/.venv/bin/python" -m unittest \
  tests.test_humaneval_benchmark \
  tests.test_humaneval_evaluator \
  tests.test_humaneval_compare

echo "HumanEval execution environment is ready."
echo "Docker image: $IMAGE"
echo "Image ID: $(docker image inspect "$IMAGE" --format '{{.Id}}')"
echo "Download data next: ./scripts/download_humaneval.sh"
