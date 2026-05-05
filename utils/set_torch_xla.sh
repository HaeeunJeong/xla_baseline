#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "pytorch" ]; then
    git clone https://github.com/pytorch/pytorch.git
fi
cd pytorch           # xla_baseline/utils/pytorch
git checkout 0fabc3b # 2.9.0

if [ ! -d "xla" ]; then
    git clone https://github.com/pytorch/xla.git
fi
cd xla               # xla_baseline/utils/pytorch/xla
git checkout 5fab705 # 2.9.8-rc3

cd ../../            # xla_baseline/utils
if [ ! -d "vision" ]; then
    git clone https://github.com/pytorch/vision.git
fi
cd vision            # xla_baseline/utils/vision
git checkout 7a9db90 # 0.24.0
cd ../               # xla_baseline/utils


# Build
conda env create --file ../env/torch-xla/environment.yaml || true

# Set up automatic LD_LIBRARY_PATH injection for torch-xla environment
CONDA_BASE="$(conda info --base)"
ENV_DIR="$CONDA_BASE/envs/torch-xla"

mkdir -p "$ENV_DIR/etc/conda/activate.d"
mkdir -p "$ENV_DIR/etc/conda/deactivate.d"

cat << 'EOF' > "$ENV_DIR/etc/conda/activate.d/env_vars.sh"
#!/bin/sh
export OLD_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.13/site-packages/torch/lib:${LD_LIBRARY_PATH}"
EOF

cat << 'EOF' > "$ENV_DIR/etc/conda/deactivate.d/env_vars.sh"
#!/bin/sh
export LD_LIBRARY_PATH="$OLD_LD_LIBRARY_PATH"
unset OLD_LD_LIBRARY_PATH
EOF
cd pytorch
conda run -n torch-xla --no-capture-output pip install -r requirements-build.txt
git submodule update --init --recursive
conda run -n torch-xla --no-capture-output python setup.py bdist_wheel
conda run -n torch-xla --no-capture-output pip install dist/torch-*.whl

cd ../vision
conda run -n torch-xla --no-capture-output python setup.py bdist_wheel
conda run -n torch-xla --no-capture-output pip install dist/torchvision-*.whl

# Create temporary swap to build torch-xla
# torch-xla build requires maximum 170GB of memory
# Skip this step if running inside Docker (swapon is not permitted in standard containers)
if [ ! -f /.dockerenv ]; then
    sudo fallocate -l 64G /swapfile2
    sudo chmod 600 /swapfile2
    sudo mkswap /swapfile2
    sudo swapon /swapfile2
fi

sudo apt install apt-transport-https curl gnupg -y
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg
sudo mv bazel-archive-keyring.gpg /usr/share/keyrings
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
sudo apt update && sudo apt install bazel-7.4.1 -y

cd ../pytorch/xla
conda run -n torch-xla --no-capture-output python setup.py bdist_wheel
conda run -n torch-xla --no-capture-output pip install dist/*.whl

# Delete swap
if [ ! -f /.dockerenv ]; then
    sudo swapoff /swapfile2
    sudo rm /swapfile2
fi
