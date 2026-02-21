#!/bin/bash

OUTPUT_DIR="/workspace/FORInstanceV2/processed"
mkdir -p "$OUTPUT_DIR"

download_if_missing() {
    local file_id="$1"
    local output="$2"
    if [ -f "$output" ]; then
        echo "Already exists, skipping: $output"
    else
        echo "Downloading: $output"
        gdown "$file_id" -O "$output"
    fi
}

download_if_missing 19t6H-Uk9I7HeZ0zhbU9S_MrJlN0J7-xZ "$OUTPUT_DIR/las_train_0.020_30000_var.joblib"
download_if_missing 1AXC7Pzvs31rnpYR37MBkCLgNPEROmIfI "$OUTPUT_DIR/las_val_0.020_30000_var.joblib"
