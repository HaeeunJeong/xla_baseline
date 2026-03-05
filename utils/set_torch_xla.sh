this version is 

git clone https://github.com/pytorch/pytorch.git
cd pytorch
git checkout 0fabc3b

git clone git clone https://github.com/pytorch/xla.git
cd xla
git checkout 5fab705 

cd ../../
git clone https://github.com/pytorch/vision.git
cd vision
git checkout 7a9db90



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

# Delete swap
sudo swapoff /swapfile2
sudo rm /swapfile2

cd ../pytorch/xla
python setup.py bdist_wheel
pip install dist/*.whl