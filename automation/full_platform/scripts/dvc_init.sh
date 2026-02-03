#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${DVC_REMOTE_URL:-}" ]]; then
  echo "Set DVC_REMOTE_URL to your s3:// bucket"
  exit 1
fi

if [[ ! -d .git ]]; then
  git init
fi

if [[ ! -d .dvc ]]; then
  dvc init
fi

dvc remote add -f s3remote "$DVC_REMOTE_URL"

dvc add data/intel data/fruits

git add data/*.dvc .dvc .gitignore

echo "DVC initialized. Run: dvc push"
