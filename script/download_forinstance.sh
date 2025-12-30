#!/usr/bin/env bash
# Download ForInstanceV2 dataset from Google Drive

echo "------------------------------------------------------------"
echo "          Downloading ForInstanceV2 Dataset"
echo "------------------------------------------------------------"
echo ""

# Color output functions
log_info() {
  echo -e "\e[34m[INFO]\e[0m: $1"
}

log_success() {
  echo -e "\e[32m[SUCCESS]\e[0m: $1"
}

log_error() {
  echo -e "\e[31m[ERROR]\e[0m: $1"
}

# Configuration
FILE_ID="16ZznV66rZYGLtOiaH__Ljt0H1Axk6jMN"
OUTPUT_DIR="data/FORInstanceV2"
OUTPUT_FILE="forinstancev2.zip"
EXTRACT_DIR="raw"

# Create data directory
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

# Function to download using gdown (recommended for Google Drive)
download_with_gdown() {
  log_info "Attempting download using gdown..."

  if ! command -v gdown &> /dev/null; then
    log_info "gdown not found. Installing via pip..."
    pip install gdown
    if [ $? -ne 0 ]; then
      log_error "Failed to install gdown"
      return 1
    fi
  fi

  log_info "Downloading file (this may take a while)..."
  gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "$OUTPUT_FILE"

  return $?
}

# Function to download using curl (fallback)
download_with_curl() {
  log_info "Attempting download using curl..."

  # Try to get the file
  curl -c /tmp/cookies.txt "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /tmp/intermezzo.html

  # Check if confirmation is needed (for large files)
  if grep -q "download_warning" /tmp/intermezzo.html; then
    log_info "Large file detected, getting confirmation code..."
    CODE=$(awk '/download_warning/ {print $NF}' /tmp/intermezzo.html | grep -o 'confirm=[^&]*' | cut -d'=' -f2)
    curl -Lb /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}" -o "$OUTPUT_FILE"
  else
    mv /tmp/intermezzo.html "$OUTPUT_FILE"
  fi

  # Cleanup
  rm -f /tmp/cookies.txt /tmp/intermezzo.html

  return $?
}

# Try gdown first, fallback to curl
if download_with_gdown; then
  log_success "Download completed successfully!"
else
  log_info "gdown failed, trying curl..."
  if download_with_curl; then
    log_success "Download completed successfully!"
  else
    log_error "Download failed with both methods"
    exit 1
  fi
fi

# Verify the download
if [ ! -f "$OUTPUT_FILE" ]; then
  log_error "Downloaded file not found!"
  exit 1
fi

FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
log_info "Downloaded file size: $FILE_SIZE"

# Check if file is too small (likely an error page)
MIN_SIZE=1000000  # 1MB in bytes
ACTUAL_SIZE=$(stat -f%z "$OUTPUT_FILE" 2>/dev/null || stat -c%s "$OUTPUT_FILE" 2>/dev/null)

if [ "$ACTUAL_SIZE" -lt "$MIN_SIZE" ]; then
  log_error "Downloaded file is too small ($FILE_SIZE). Download may have failed."
  log_error "Please check the file manually: $OUTPUT_DIR/$OUTPUT_FILE"
  exit 1
fi

# Extract if it's a zip file
log_info "Checking file type..."
if file "$OUTPUT_FILE" | grep -q "Zip archive"; then
  log_info "Extracting archive..."

  if ! command -v unzip &> /dev/null; then
    log_error "unzip not found. Please install it to extract the archive."
    log_info "You can manually extract: $OUTPUT_DIR/$OUTPUT_FILE"
    exit 1
  fi

  unzip -q "$OUTPUT_FILE"

  if [ $? -eq 0 ]; then
    log_success "Archive extracted successfully!"
    log_info "Cleaning up zip file..."
    rm "$OUTPUT_FILE"
    log_success "Dataset ready at: $OUTPUT_DIR/"
  else
    log_error "Failed to extract archive"
    log_info "You can manually extract: $OUTPUT_DIR/$OUTPUT_FILE"
    exit 1
  fi
else
  log_info "File is not a zip archive, keeping as is: $OUTPUT_FILE"
fi

echo ""
echo "------------------------------------------------------------"
log_success "ForInstanceV2 dataset download complete!"
echo "------------------------------------------------------------"