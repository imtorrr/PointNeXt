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
# Install and configure Ninja build system for faster C++ builds
log_info "Setting up Ninja build system for faster C++ compilation..."
if ! command -v ninja &> /dev/null; then
  uv pip install ninja
  log_success "Ninja build system installed"
else
  log_info "Ninja build system already available"
fi
export CMAKE_GENERATOR=Ninja
log_info "Configured to use Ninja for C++ extension builds"

echo ""
log_info "Building C++ extensions in parallel for faster compilation..."
log_info "Using $MAX_JOBS parallel jobs with Ninja build system"
echo ""

# Status file for tracking progress
STATUS_FILE="/tmp/build_status_$$.txt"
> "$STATUS_FILE"

# List of extensions to build
declare -A extensions=(
  ["pointnet2_batch"]="required"
  ["pointops"]="optional"
  ["chamfer_dist"]="optional"
  ["emd"]="optional"
  ["subsampling"]="optional"
)

# Print initial status
echo "╔════════════════════════════════════════════════════════════╗"
echo "║              C++ Extension Build Progress                  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
for name in "${!extensions[@]}"; do
  echo "  ⏳ $name ... pending"
done
echo ""

# Function to build a C++ extension
build_extension() {
  local name=$1
  local path=$2
  local required=$3
  local log_file="/tmp/build_${name}.log"
  local status_file=$4

  # Mark as in progress
  echo "BUILDING:$name" >> "$status_file"

  cd "$path" 2>&1 > "$log_file"
  if python setup.py install >> "$log_file" 2>&1; then
    echo "SUCCESS:$name" >> "$status_file"
  else
    echo "FAILED:$name:$required:$log_file" >> "$status_file"
  fi
}

export -f build_extension
export STATUS_FILE
export CMAKE_GENERATOR

# Start monitoring progress in background
(
  prev_count=0
  while true; do
    if [ ! -f "$STATUS_FILE" ]; then
      sleep 0.1
      continue
    fi

    # Count completed builds
    completed=$(grep -c "SUCCESS:\|FAILED:" "$STATUS_FILE" 2>/dev/null || echo 0)
    completed=${completed:-0}
    building=$(grep -c "BUILDING:" "$STATUS_FILE" 2>/dev/null || echo 0)
    building=${building:-0}
    total=${#extensions[@]}

    if [ $completed -ne $prev_count ] || [ $building -gt 0 ]; then
      # Move cursor up to redraw status
      if [ $prev_count -gt 0 ]; then
        tput cuu $((${#extensions[@]} + 1))
      fi

      # Display current status
      for name in "${!extensions[@]}"; do
        if grep -q "SUCCESS:$name" "$STATUS_FILE" 2>/dev/null; then
          echo -e "  \e[32m✓\e[0m $name ... \e[32mcomplete\e[0m          "
        elif grep -q "FAILED:$name" "$STATUS_FILE" 2>/dev/null; then
          echo -e "  \e[31m✗\e[0m $name ... \e[31mfailed\e[0m            "
        elif grep -q "BUILDING:$name" "$STATUS_FILE" 2>/dev/null; then
          echo -e "  \e[33m⚙\e[0m  $name ... \e[33mbuilding\e[0m          "
        else
          echo "  ⏳ $name ... pending          "
        fi
      done

      # Progress bar
      percent=0
      if [ "$total" -gt 0 ]; then
        percent=$((completed * 100 / total))
      fi
      bar_width=40
      filled=$((percent * bar_width / 100))
      printf "\n  ["
      printf "%${filled}s" | tr ' ' '='
      printf "%$((bar_width - filled))s" | tr ' ' ' '
      printf "] %d/%d (%d%%)\n" $completed $total $percent

      prev_count=$completed
    fi

    # Exit when all complete
    if [ $completed -eq $total ]; then
      break
    fi

    sleep 0.2
  done
) &
monitor_pid=$!

# Build all extensions in parallel
cd openpoints/cpp
build_extension "pointnet2_batch" "pointnet2_batch" "required" "$STATUS_FILE" &
build_extension "pointops" "pointops" "optional" "$STATUS_FILE" &
build_extension "chamfer_dist" "chamfer_dist" "optional" "$STATUS_FILE" &
build_extension "emd" "emd" "optional" "$STATUS_FILE" &
build_extension "subsampling" "subsampling" "optional" "$STATUS_FILE" &
wait

# Wait for monitor to finish
wait $monitor_pid 2>/dev/null

cd ../../..
echo ""
echo ""

# Process final results
has_errors=false
while IFS= read -r line; do
  if [[ $line == FAILED:* ]]; then
    IFS=':' read -r _ name required log_file <<< "$line"
    if [[ $required == "required" ]]; then
      log_error "Failed to install $name. Check $log_file for details."
      has_errors=true
    else
      log_warning "Failed to install $name. This is optional. Check $log_file for details."
    fi
  fi
done < "$STATUS_FILE"

# Cleanup
rm -f "$STATUS_FILE"

if [ "$has_errors" = true ]; then
  log_error "Required C++ extensions failed to build. Please check the logs."
  exit 1
fi

log_success "All C++ extensions built successfully!"

echo ""
log_info "Environment setup complete!"
echo "------------------------------------------------------------"
