import copy
from typing import *

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from modules.models.autoencoder import Encoder
from modules.models.unet import Unet


# from .attention import Attention


def split_array_into_overlapping_patches(arr, patch_size, stride):
    # Get the array's shape
    batch_size, height, width, num_channels = arr.shape
    num_patches_vertical = (height - patch_size) // stride + 1
    num_patches_horizontal = (width - patch_size) // stride + 1

    # Create an array of indices for extracting patches
    y_indices = stride * jnp.arange(num_patches_vertical)
    x_indices = stride * jnp.arange(num_patches_horizontal)
    yy, xx = jnp.meshgrid(y_indices, x_indices)
    yy = yy.reshape(-1, 1)
    xx = xx.reshape(-1, 1)

    # Calculate the indices for patches extraction
    y_indices = yy + jnp.arange(patch_size)
    x_indices = xx + jnp.arange(patch_size)

    # Extract the patches using advanced indexing
    patches = arr[:, y_indices[:, :, None], x_indices[:, None, :]]

    return patches


"""
 x_self_cond = nn.Conv(3 * n ** 2, (5, 5), padding="SAME", dtype=self.dtype)(encoder_output)
                x_self_cond = einops.rearrange(x_self_cond, 'b h w (c p1 p2)->b (h p1) (w p2) c', p1=n, p2=n)
                x_self_cond = jax.image.resize(x_self_cond, x.shape, 'bicubic')
        # b,h,w,c=x.shape
        # x = split_array_into_overlapping_patches(x,h//self.patch_size,h//self.patch_size//2)
        # x=einops.rearrange(x,'b n h w c ->b w h (n c)')
        # print(x.shape)

  self.decoder_latent_1d = nn.Sequential([
            # nn.GroupNorm(num_groups=min(8, latent.shape[-1])),
            nn.silu,
            nn.Conv(512, (1, 1)),
            GlobalAveragePool(),
            Rearrange('b h w c->b (h w c)'),
            nn.Dense(512)
        ])


"""


class DiffEncoder(nn.Module):
    dim: int = 64
    dim_mults: Sequence = (1, 2, 4, 4)
    num_res_blocks: Any = 2
    out_channels: int = 3
    resnet_block_groups: int = 8,
    channels: int = 3,
    dtype: Any = jnp.bfloat16
    encoder_configs: Any = None
    encoder_type: str = '2D'
    res_type: Any = 'default'
    latent_type: Any = 'tanh'
    patch_size: int = 1
    time_embedding: bool = False

    def setup(self):
        encoder_configs = flax.core.frozen_dict.unfreeze(copy.deepcopy(self.encoder_configs))

        if self.latent_type == 'double_z':
            encoder_configs['latent'] = encoder_configs['latent'] * 2

        self.encoder = Encoder(encoder_type=self.encoder_type, **encoder_configs, name='Encoder')
        self.unet = Unet(dim=self.dim,
                         dim_mults=self.dim_mults,
                         num_res_blocks=self.num_res_blocks,
                         out_channels=self.out_channels,
                         channels=self.channels,
                         dtype=self.dtype,
                         encoder_type=self.encoder_type,
                         res_type=self.res_type,
                         use_encoder=True,
                         patch_size=self.patch_size,
                         n=(len(encoder_configs['dims']) - 1) ** 2,
                         time_embedding=self.time_embedding
                         )

    def encode(self, x, z_rng=None, *args, **kwargs):
        print(f'encoder input:{x.shape}')
        x = self.encoder(x)
        if self.latent_type == 'tanh':
            x = nn.tanh(x)
        elif self.latent_type == 'sin':
            x = jnp.sin(x)
        elif self.latent_type == 'double_z':
            if z_rng is None:
                print('z_rng is None ,z_rng will default as 42')
                z_rng = jax.random.PRNGKey(42)

            mean, log_var = jnp.split(x, 2, -1)
            mean = mean.clip(-1, 1)
            log_var = log_var.clip(-1, 1)
            self.sow('intermediates', 'mean', mean)
            self.sow('intermediates', 'log_var', log_var)
            x = self.reparameter(z_rng, mean, log_var)
        elif self.latent_type == 'double_z_tanh':
            if z_rng is None:
                print('z_rng is None ,z_rng will default as 42')
                z_rng = jax.random.PRNGKey(42)

            mean, log_var = jnp.split(x, 2, -1)
            # mean = mean.clip(-3, 3)
            log_var = log_var.clip(-20, 20)
            self.sow('intermediates', 'mean', mean)
            self.sow('intermediates', 'log_var', log_var)
            x = self.reparameter(z_rng, mean, log_var)
            x = nn.tanh(x)

        elif self.latent_type == 'clip':
            x = jnp.clip(x, -1, 1)
        return x

    def decode(self, x, time, latent=None, z_rng=None, *args, **kwargs):
        x = self.unet(x, time, latent, z_rng=z_rng)
        return x

    def __call__(self, x, time, x_self_cond=None, z_rng=None, *args, **kwargs):
        latent = self.encode(x_self_cond, z_rng=z_rng)
        x = self.decode(x, time, latent, z_rng=z_rng)
        return x

    def reparameter(self, rng, mean, logvar):
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(rng, logvar.shape)
        return mean + eps * std


encode = DiffEncoder.encode
decode = DiffEncoder.decode
