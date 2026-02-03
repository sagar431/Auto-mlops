#!/usr/bin/env bash
set -euo pipefail

OUTPUT_FILE=${1:-reports/cluster_outputs.md}
NAMESPACE=${2:-auto-mlops}

mkdir -p "$(dirname "$OUTPUT_FILE")"

{
  echo "# Cluster Output Capture"
  echo
  echo "## kubectl describe <deployment>"
  echo
  echo '```'
  kubectl -n "$NAMESPACE" describe deploy/model-api || true
  echo '```'
  echo
  echo "## kubectl describe <pod>"
  echo
  echo '```'
  kubectl -n "$NAMESPACE" describe pod -l app=model-api || \\
    kubectl -n "$NAMESPACE" describe pod -l app.kubernetes.io/component=model-api || true
  echo '```'
  echo
  echo "## kubectl describe <ingress>"
  echo
  echo '```'
  kubectl -n "$NAMESPACE" describe ingress auto-mlops || true
  echo '```'
  echo
  echo "## kubectl top pod"
  echo
  echo '```'
  kubectl -n "$NAMESPACE" top pod || true
  echo '```'
  echo
  echo "## kubectl top node"
  echo
  echo '```'
  kubectl top node || true
  echo '```'
  echo
  echo "## kubectl get all -A -o yaml"
  echo
  echo '```'
  kubectl get all -A -o yaml || true
  echo '```'
} > "$OUTPUT_FILE"

printf "Saved %s\n" "$OUTPUT_FILE"
