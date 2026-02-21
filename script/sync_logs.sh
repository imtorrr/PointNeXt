#!/bin/bash

REGION="eu-ro-1"
ENDPOINT_URL="https://s3api-eu-ro-1.runpod.io"
BUCKET="s3://ec99o9fb22"
LOCAL_DIR="/workspace/log"

mkdir -p "$LOCAL_DIR"

echo "Syncing $LOCAL_DIR -> $BUCKET/log ..."
aws s3 sync "$LOCAL_DIR" "$BUCKET/log" \
  --region "$REGION" \
  --endpoint-url "$ENDPOINT_URL"

echo "Sync complete."
