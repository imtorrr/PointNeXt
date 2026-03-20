#!/bin/bash

OUTPUT_DIR="/workspace/FieldData/processed"
mkdir -p "$OUTPUT_DIR"

download_if_missing() {
  local file_id="$1"
  local output="$2"
  if [ -f "$output" ]; then
    echo "Already exists, skipping: $output"
  else
    echo "Downloading: $output"
    uv run gdown "$file_id" -O "$output"
  fi
}

download_if_missing 1O2X0d2ucX6z2zeytSL8CQCaCS8heZfz2 "$OUTPUT_DIR/las_train_0.020_30000_var.joblib"
download_if_missing 1ILmN5JFwKBS4hVpHz4riCgXtxEgoyknq "$OUTPUT_DIR/las_val_0.020_30000_var.joblib"
