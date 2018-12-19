#!/bin/bash -eu
export DRIVER_INSTALLER_FILE_NAME="driver_installer.run"
#wget http://us.download.nvidia.com/tesla/410.72/NVIDIA-Linux-x86_64-410.72.run -O ${DRIVER_INSTALLER_FILE_NAME}
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run -O ${DRIVER_INSTALLER_FILE_NAME}
chmod +x ${DRIVER_INSTALLER_FILE_NAME}
sudo ./${DRIVER_INSTALLER_FILE_NAME} --silent --no-drm
rm -rf ${DRIVER_INSTALLER_FILE_NAME}

wget https://www.dropbox.com/s/c80oz0akb2j19kv/libcudnn7_7.0.5.15-1%2Bcuda9.0_amd64.deb?dl=1 -O libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb
wget https://www.dropbox.com/s/3qqyjtyuj7hhskq/libcudnn7-dev_7.0.5.15-1%2Bcuda9.0_amd64.deb?dl=1 -O libcudnn7-dev_7.0.5.15-1+cuda9.0_amd64.deb
wget https://www.dropbox.com/s/kxxyqeo8lon9vx8/libcudnn7-doc_7.0.5.15-1%2Bcuda9.0_amd64.deb?dl=1 -O libcudnn7-doc_7.0.5.15-1+cuda9.0_amd64.deb
sudo dpkg -i libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb
sudo dpkg -i libcudnn7-dev_7.0.5.15-1+cuda9.0_amd64.deb
sudo dpkg -i libcudnn7-doc_7.0.5.15-1+cuda9.0_amd64.deb

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"

git clone https://github.com/nhanvtran/MachineLearningNotebooks.git -b jmgd/bwcustomweights
cd MachineLearningNotebooks
conda create -n myamlenv Python=3.6 cython numpy
source activate myamlenv
conda install -y matplotlib tqdm scikit-learn
pip install --upgrade azureml-sdk[notebooks,automl] azureml-dataprep
pip install --upgrade azureml-sdk[contrib]
pip install tables
pip install jupyterhub
pip install --ignore-installed tensorflow_gpu==1.10

mkdir -p data
mkdir -p models
mkdir -p weights

# jupyter notebook --port 5000 --ip 0.0.0.0 --no-browser
exit 0
