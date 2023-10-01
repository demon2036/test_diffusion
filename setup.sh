apt update
apt install unrar rar
apt-get -y --force-yes install golang
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install flax
pip install albumentations einops tqdm matplotlib jax-smi pytorch_fid
pip install tensorflow==2.13 tensorflow-datasets webdataset keras-cv