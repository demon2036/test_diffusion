import math
from functools import partial
import einops
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import *
from einops.layers.flax import Rearrange
import optax
from modules.models.nafnet import NAFBlock
from modules.models.autoencoder import Encoder, AutoEncoderKL, AutoEncoder
from modules.models.transformer import Transformer
from modules.models.embedding import SinusoidalPosEmb
from modules.models.resnet import ResBlock, DownSample, UpSample, EfficientBlock, GlobalAveragePool


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

"""


class Encoder2DLatent(nn.Module):
    n: int = 8
    dtype: Any = 'bfloat16'
    shape: Any = None

    @nn.compact
    def __call__(self, latent, *args, **kwargs):
        n = self.n
        x_self_cond = nn.Conv(3 * n ** 2, (5, 5), padding="SAME", dtype=self.dtype)(latent)
        x_self_cond = einops.rearrange(x_self_cond, 'b h w (c p1 p2)->b (h p1) (w p2) c', p1=n, p2=n)
        x_self_cond = jax.image.resize(x_self_cond, self.shape, 'bicubic')
        return x_self_cond


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

    def setup(self):

        encoder_configs = self.encoder_configs
        self.encoder = Encoder(encoder_type=self.encoder_type,**encoder_configs, name='Encoder')
        self.decoder_latent_1d = nn.Sequential([
            # nn.GroupNorm(num_groups=min(8, latent.shape[-1])),
            nn.silu,
            nn.Conv(512, (1, 1)),
            GlobalAveragePool(),
            Rearrange('b h w c->b (h w c)'),
            nn.Dense(512)
        ])

    def encode(self, x, *args, **kwargs):
        x = self.encoder(x)
        x = nn.tanh(x)
        return x

    @nn.compact
    def decode(self, x, time, latent=None, z_rng=None, *args, **kwargs):
        if type(self.num_res_blocks) == int:
            num_res_blocks = (self.num_res_blocks,) * len(self.dim_mults)
        else:
            assert len(self.num_res_blocks) == len(self.dim_mults)
            num_res_blocks = self.num_res_blocks

        # Define ResBlock Type

        if self.res_type == 'default':
            res_block = ResBlock
        elif self.res_type == "NAF":
            res_block = NAFBlock
        elif self.res_type == "efficient":
            res_block = EfficientBlock
        else:
            raise NotImplementedError()

        assert self.encoder_type in ['1D', '2D', 'Both']
        print(f'latent shape:{latent.shape}')
        if self.encoder_type == '1D':
            cond_emb = latent
            x_self_cond = None
        elif self.encoder_type == '2D':
            cond_emb = None
            x_self_cond = Encoder2DLatent(shape=x.shape)(latent)
        elif self.encoder_type == 'Both':
            cond_emb = self.decoder_latent_1d(latent)
            x_self_cond = Encoder2DLatent(shape=x.shape)(latent)

        if x_self_cond is not None:
            x = jnp.concatenate([x, x_self_cond], axis=3)

        print(x.shape)
        time_dim = self.dim * 4
        t = nn.Sequential([
            SinusoidalPosEmb(self.dim),
            nn.Dense(time_dim, dtype=self.dtype),
            nn.gelu,
            nn.Dense(time_dim, dtype=self.dtype)
        ])(time)

        x = nn.Conv(self.dim, (7, 7), (1, 1), padding="SAME", dtype=self.dtype)(x)

        h = [x]

        for i, (dim_mul, num_res_block) in enumerate(zip(self.dim_mults, num_res_blocks)):
            dim = self.dim * dim_mul
            for _ in range(num_res_block):
                x = res_block(dim, dtype=self.dtype)(x, t, cond_emb)
                h.append(x)

            if i != len(self.dim_mults) - 1:
                x = DownSample(dim, dtype=self.dtype)(x)
                h.append(x)

        x = res_block(dim, dtype=self.dtype)(x, t, cond_emb)
        x = res_block(dim, dtype=self.dtype)(x, t, cond_emb)

        reversed_dim_mults = list(reversed(self.dim_mults))
        reversed_num_res_blocks = list(reversed(num_res_blocks))

        for i, (dim_mul, num_res_block) in enumerate(zip(reversed_dim_mults, reversed_num_res_blocks)):
            dim = self.dim * dim_mul
            for _ in range(num_res_block + 1):
                x = jnp.concatenate([x, h.pop()], axis=3)
                x = res_block(dim, dtype=self.dtype)(x, t, cond_emb)

            if i != len(self.dim_mults) - 1:
                x = UpSample(dim, dtype=self.dtype)(x)

        x = nn.GroupNorm()(x)
        x = nn.silu(x)
        x = nn.Conv(self.out_channels, (3, 3), dtype="float32")(x)

        return x

    def __call__(self, x, time, x_self_cond=None, z_rng=None, *args, **kwargs):
        latent = self.encode(x_self_cond)
        x = self.decode(x, time, latent)
        return x
