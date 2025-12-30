#!/usr/bin/env bash
# command to install this environment: source init.sh

echo "------------------------------------------------------------"
echo "               Setting up the coding environment..."
echo "------------------------------------------------------------"
echo ""

# Function to display messages with a prefix
log_info() {
  echo -e "\e[34m[INFO]\e[0m: $1"
}

log_success() {
  echo -e "\e[32m[SUCCESS]\e[0m: $1"
}

log_warning() {
  echo -e "\e[33m[WARNING]\e[0m: $1"
}

log_error() {
  echo -e "\e[31m[ERROR]\e[0m: $1"
}

# Optimize CUDA architectures for faster compilation (only common modern GPUs)
# Adjust this list based on your target GPUs
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0"

# Enable parallel compilation - use all available CPU cores
export MAX_JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 5)

# Enable ccache if available for faster rebuilds
if command -v ccache &> /dev/null; then
  export CC="ccache gcc"
  export CXX="ccache g++"
  export CUDA_NVCC_EXECUTABLE="ccache nvcc"
  log_info "ccache detected - build caching enabled for faster rebuilds"
fi

echo ""
log_info "Updating git submodules..."
git submodule update --init --recursive
if [ $? -eq 0 ]; then
  log_success "Git submodules initialized and updated."
else
  log_error "Failed to initialize/update git submodules."
fi

git submodule update --remote --merge
if [ $? -eq 0 ]; then
  log_success "Git submodules updated to the latest version and merged."
else
  log_warning "Could not update all git submodules to the latest version."
fi

echo ""
log_info "Creating virtual environment using uv..."
uv venv --python 3.12
source .venv/bin/activate
log_info "Installing dependencies from requirements.txt using uv..."
uv pip install -r requirements.txt
if [ $? -eq 0 ]; then
  log_success "Dependencies installed successfully!"
else
  log_error "Failed to install dependencies."
fi

echo ""
log_info "Building C++ extensions in parallel for faster compilation..."
log_info "Using $MAX_JOBS parallel jobs"

# Function to build a C++ extension
build_extension() {
  local name=$1
  local path=$2
  local required=$3
  local log_file="/tmp/build_${name}.log"

  cd "$path" 2>&1 > "$log_file"
  if pip install -e . --no-build-isolation >> "$log_file" 2>&1; then
    echo "SUCCESS:$name"
  else
    echo "FAILED:$name:$required:$log_file"
  fi
}

export -f build_extension
export -f log_success
export -f log_warning
export -f log_error

# Build all extensions in parallel
cd openpoints/cpp
results=$(
  build_extension "pointnet2_batch" "pointnet2_batch" "required" &
  build_extension "pointops" "pointops" "optional" &
  build_extension "chamfer_dist" "chamfer_dist" "optional" &
  build_extension "emd" "emd" "optional" &
  build_extension "subsampling" "subsampling", "optional" &
  wait
)

# Process results
cd ../../..
echo ""
has_errors=false

while IFS= read -r line; do
  if [[ $line == SUCCESS:* ]]; then
    name="${line#SUCCESS:}"
    log_success "$name installed successfully!"
  elif [[ $line == FAILED:* ]]; then
    IFS=':' read -r _ name required log_file <<< "$line"
    if [[ $required == "required" ]]; then
      log_error "Failed to install $name. Check $log_file for details."
      has_errors=true
    else
      log_warning "Failed to install $name. This is optional. Check $log_file for details."
    fi
  fi
done <<< "$results"

if [ "$has_errors" = true ]; then
  log_error "Required C++ extensions failed to build. Please check the logs."
  exit 1
fi

echo ""
log_info "Environment setup complete!"
deactivate
echo "------------------------------------------------------------"
