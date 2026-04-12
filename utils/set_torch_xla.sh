git clone https://github.com/pytorch/pytorch.git
cd pytorch
git checkout 0fabc3b # 2.9.0

git clone https://github.com/pytorch/xla.git
cd xla
git checkout 5fab705 # 2.9.8-rc3

cd ../../
git clone https://github.com/pytorch/vision.git
cd vision
git checkout 7a9db90 # 0.24.0



# Build
conda env create --file ../env/torch-xla/environment.yaml
conda activate torch-xla

cd ../pytorch
pip install -r requirements-build.txt
git submodule update --init --recursive
python setup.py bdist_wheel
pip install dist/torch-*.whl

cd ../vision
python setup.py bdist_wheel
pip install dist/torchvision-*.whl

# Create temporary swap to build torch-xla
# torch-xla build requires maximum 170GB of memory
sudo fallocate -l 64G /swapfile2
sudo chmod 600 /swapfile2
sudo mkswap /swapfile2
sudo swapon /swapfile2

sudo apt install apt-transport-https curl gnupg -y
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg
sudo mv bazel-archive-keyring.gpg /usr/share/keyrings
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
sudo apt update && sudo apt install bazel-7.4.1

cd ../pytorch/xla
python setup.py bdist_wheel
pip install dist/*.whl

# Delete swap
sudo swapoff /swapfile2
sudo rm /swapfile2
