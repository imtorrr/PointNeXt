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

export TORCH_CUDA_ARCH_LIST="5.0;6.0;7.0;7.5;8.0;8.6;9.0"

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
log_info "Installing cpp extensions for pointnet++ library..."
cd openpoints/cpp/pointnet2_batch
python setup.py install
if [ $? -eq 0 ]; then
  log_success "pointnet2_batch cpp extensions installed successfully!"
else
  log_error "Failed to install pointnet2_batch cpp extensions. Check the output for errors."
fi
cd ..

echo ""
log_info "Installing pointops library (for Point Transformer and Stratified Transformer)..."
cd pointops/
python setup.py install
if [ $? -eq 0 ]; then
  log_success "pointops library installed successfully!"
else
  log_warning "Failed to install pointops library. This is only necessary if you need Point Transformer and Stratified Transformer."
fi
cd ..

echo ""
log_info "Installing chamfer_dist library (for reconstruction tasks)..."
cd chamfer_dist/
python setup.py install
if [ $? -eq 0 ]; then
  log_success "chamfer_dist library installed successfully!"
else
  log_warning "Failed to install chamfer_dist library. This is only necessary if you are interested in reconstruction tasks."
fi
cd ..

echo ""
log_info "Installing emd library (for reconstruction tasks)..."
cd emd/
python setup.py install
if [ $? -eq 0 ]; then
  log_success "emd library installed successfully!"
else
  log_warning "Failed to install emd library. This is only necessary if you are interested in reconstruction tasks."
fi
cd ../../../

echo ""
log_info "Environment setup complete!"
deactivate
echo "------------------------------------------------------------"
