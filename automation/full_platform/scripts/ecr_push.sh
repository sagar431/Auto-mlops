#!/usr/bin/env bash
set -euo pipefail

REGION=${AWS_REGION:-us-east-1}
PROFILE=${AWS_PROFILE:-default}
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --profile "$PROFILE")
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

MODEL_REPO=${MODEL_REPO:-auto-mlops-model-api}
WEB_REPO=${WEB_REPO:-auto-mlops-web-ui}
TAG=${TAG:-latest}

aws ecr describe-repositories --repository-names "$MODEL_REPO" --profile "$PROFILE" --region "$REGION" >/dev/null 2>&1 || \
  aws ecr create-repository --repository-name "$MODEL_REPO" --profile "$PROFILE" --region "$REGION"

aws ecr describe-repositories --repository-names "$WEB_REPO" --profile "$PROFILE" --region "$REGION" >/dev/null 2>&1 || \
  aws ecr create-repository --repository-name "$WEB_REPO" --profile "$PROFILE" --region "$REGION"

aws ecr get-login-password --region "$REGION" --profile "$PROFILE" | \
  docker login --username AWS --password-stdin "$ECR_URI"

# Build and tag images

docker build -t "$MODEL_REPO:$TAG" -f services/model_api/Dockerfile services/model_api

docker build -t "$WEB_REPO:$TAG" -f services/web_ui/Dockerfile services/web_ui

# Tag for ECR

docker tag "$MODEL_REPO:$TAG" "$ECR_URI/$MODEL_REPO:$TAG"

docker tag "$WEB_REPO:$TAG" "$ECR_URI/$WEB_REPO:$TAG"

# Push

docker push "$ECR_URI/$MODEL_REPO:$TAG"

docker push "$ECR_URI/$WEB_REPO:$TAG"

printf "Pushed:\n  %s/%s:%s\n  %s/%s:%s\n" "$ECR_URI" "$MODEL_REPO" "$TAG" "$ECR_URI" "$WEB_REPO" "$TAG"
