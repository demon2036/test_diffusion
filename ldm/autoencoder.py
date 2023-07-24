import argparse
import os
from resnet import Block, EfficientBlock, EfficientBlock2
import torch
from flax.training.common_utils import shard_prng_key, shard
from tqdm import tqdm
import einops
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import dynamic_scale as dynamic_scale_lib, train_state, orbax_utils
from dataset import generator
from typing import *
from functools import partial
import optax
from utils import *
from loss import l1_loss, l2_loss

os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'




class DownBlock(nn.Module):
    dim: int
    dtype: str

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        x = einops.rearrange(x, 'b  (h p1) (w p2) c ->b h w (c p1 p2)', p1=2, p2=2)
        x = nn.Conv(self.dim, (1, 1), dtype=self.dtype)(x)
        return x


class UpBlock(nn.Module):
    dim: int
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        b, h, w, c = x.shape
        x = jax.image.resize(x, shape=(b, h * 2, w * 2, c), method="nearest")
        x = nn.Conv(self.dim , (3, 3), padding="SAME", dtype=self.dtype)(x)
        #x = einops.rearrange(x, ' b h w (c p1 p2)->b (h p1) (w p2) c', p1=2, p2=2)
        return x


class ResBlock(nn.Module):
    dim: int
    dtype: str
    @nn.compact
    def __call__(self, x):
        b, _, _, c = x.shape
        hidden = Block(self.dim, self.dtype)(x)
        hidden = Block(self.dim, self.dtype)(hidden)

        if c != self.dim:
            x = nn.Conv(self.dim, (1, 1), dtype=self.dtype)(x)
        x=hidden + x
        return x


class DownSample(nn.Module):
    dim: int
    num_blocks: int
    add_down: bool
    dtype: str

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        for _ in range(self.num_blocks):
            x = ResBlock(self.dim, self.dtype)(x)
        if self.add_down:
            x = DownBlock(self.dim, self.dtype)(x)
        return x


class UpSample(nn.Module):
    dim: int
    num_blocks: int
    add_up: bool
    dtype: str
    @nn.compact
    def __call__(self, x, *args, **kwargs):
        for _ in range(self.num_blocks):
            x = ResBlock(self.dim, self.dtype)(x)

        if self.add_up:
            x = UpBlock(self.dim, self.dtype)(x)
        return x
class Encoder(nn.Module):
    dims: Sequence
    num_blocks: int
    dtype: str
    latent: int
    @nn.compact
    def __call__(self, x, *args, **kwargs):
        x = nn.Conv(self.dims[0], (7, 7), dtype=self.dtype,padding="same")(x)

        for i, dim in enumerate(self.dims):
            x = DownSample(dim, self.num_blocks, True if i != len(self.dims) else False,
                           dtype=self.dtype)(x)
        x = nn.Conv(self.latent, (1, 1), dtype=self.dtype)(x)
        x = nn.tanh(x)
        return x


class Decoder(nn.Module):
    dims: Sequence
    num_blocks: int
    dtype: str
    @nn.compact
    def __call__(self, x, *args, **kwargs):
        x = nn.Conv(self.dims[0], (7, 7), dtype=self.dtype, padding="same")(x)
        for i, dim in enumerate(self.dims):
            x = UpSample(dim, self.num_blocks, True if i != len(self.dims) else False,
                         dtype=self.dtype)(x)

        x = nn.Conv(3, (3, 3), dtype='float32',padding='SAME')(x)
        x = nn.tanh(x)
        return x
class AutoEncoder(nn.Module):
    dims: Sequence = (64, 128, 256)
    num_blocks: int = 2
    dtype: str = 'bfloat16'
    latent: int = 3
    @nn.compact
    def __call__(self, x, *args, **kwargs):
        x = Encoder(self.dims, self.num_blocks, self.dtype, self.latent)(x)
        reversed_dims = list(reversed(self.dims))
        x = Decoder(reversed_dims, self.num_blocks, self.dtype)(x)
        return x

def sample_save_image_autoencoder(state, save_path, steps, data):
    os.makedirs(save_path, exist_ok=True)
    @jax.pmap
    def infer(state,params, data):
        sample = state.apply_fn({'params': params}, data)
        return sample

    if steps<50000:
        sample = infer(state, state.params, data)
    else:
        sample = infer(state, state.ema_params, data)


    all_image = jnp.concatenate([sample, data], axis=1)
    all_image = all_image / 2 + 0.5
    all_image = einops.rearrange(all_image, 'n b h w c->(n b) c h w')
    all_image = np.array(all_image)
    all_image = torch.Tensor(all_image)

    save_image(all_image, f'{save_path}/{steps}.png')
