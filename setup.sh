apt update
apt install unrar rar
#apt-get -y --force-yes install golang
apt-get  install golang

pip install diffusers

pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install flax --ignore-installed PyYAML
pip install albumentations einops tqdm matplotlib jax-smi
pip install tensorflow==2.13 tensorflow-datasets webdataset keras-cv timm

#pip install pytorch_fid