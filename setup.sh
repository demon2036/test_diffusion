apt update
apt install unrar rar
apt-get -y --force-yes install golang
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install flax
pip install albumentations einops tqdm matplotlib jax-smi pytorch_fid