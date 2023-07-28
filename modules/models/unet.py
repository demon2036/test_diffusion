import math
from functools import partial
import einops
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import *

from modules.models.autoencoder import Encoder
from modules.models.transformer import Transformer
from modules.models.embedding import SinusoidalPosEmb
from modules.models.resnet import ResBlock, DownSample, UpSample


class Unet(nn.Module):
    dim: int = 64
    out_channels: int = 3
    resnet_block_groups: int = 8,
    channels: int = 3,
    dim_mults: Sequence = (1, 2, 4, 8)
    dtype: Any = jnp.bfloat16
    self_condition: bool = False
    use_encoder: bool = False
    encoder_configs: Any = None

    @nn.compact
    def __call__(self, x, time, x_self_cond=None, *args, **kwargs):

        if self.use_encoder:
            n = 2 ** 3
            x_self_cond = Encoder(**self.encoder_configs)(x_self_cond)
            x_self_cond = nn.Conv(3 * n ** 2, (5, 5), padding="SAME", dtype=self.dtype)(x_self_cond)
            x = einops.rearrange('b h w (c p1 p2)->b (h p1) (w p2) c', p1=n, p2=n)
            x_self_cond = jax.image.resize(x_self_cond, x.shape, 'bicubic')

        if x_self_cond is not None and self.self_condition:
            x = jnp.concatenate([x, x_self_cond], axis=3)
        elif self.self_condition:
            x = jnp.concatenate([x, jnp.zeros_like(x)], axis=3)
        print(x.shape)

        time_dim = self.dim * 4
        t = nn.Sequential([
            SinusoidalPosEmb(self.dim),
            nn.Dense(time_dim, dtype=self.dtype),
            nn.gelu,
            nn.Dense(time_dim, dtype=self.dtype)
        ])(time)

        x = nn.Conv(self.dim, (7, 7), padding="SAME", dtype=self.dtype)(x)
        r = x

        h = []

        for i, dim_mul in enumerate(self.dim_mults):
            dim = self.dim * dim_mul

            x = ResBlock(dim, dtype=self.dtype)(x, t)
            h.append(x)
            x = ResBlock(dim, dtype=self.dtype)(x, t)
            h.append(x)
            if i != len(self.dim_mults) - 1:
                x = DownSample(dim, dtype=self.dtype)(x)
            else:
                x = nn.Conv(dim, (3, 3), dtype=self.dtype, padding="SAME")(x)

        x = ResBlock(dim, dtype=self.dtype)(x, t)
        # x = self.mid_attn(x) + x
        x = ResBlock(dim, dtype=self.dtype)(x, t)

        reversed_dim_mults = list(reversed(self.dim_mults))

        for i, dim_mul in enumerate(reversed_dim_mults):
            dim = self.dim * dim_mul

            x = jnp.concatenate([x, h.pop()], axis=3)
            x = ResBlock(dim, dtype=self.dtype)(x, t)
            x = jnp.concatenate([x, h.pop()], axis=3)
            x = ResBlock(dim, dtype=self.dtype)(x, t)

            if i != len(self.dim_mults) - 1:
                x = UpSample(dim, dtype=self.dtype)(x)
            else:
                x = nn.Conv(dim, (3, 3), dtype=self.dtype, padding="SAME")(x)

        x = jnp.concatenate([x, r], axis=3)
        x = ResBlock(dim, dtype=self.dtype)(x, t)
        x = nn.Conv(self.out_channels, (1, 1), dtype="float32")(x)
        return x


class MultiUnet(nn.Module):
    dim: int = 32
    out_channels: int = 3
    resnet_block_groups: int = 8,
    channels: int = 3,
    dim_mults: Sequence = (1, 2, 4, 8)
    dtype: Any = jnp.bfloat16
    num_unets: int = 4
    self_condition: bool = False

    @nn.compact
    def __call__(self, x, time, x_self_cond=None, *args, **kwargs):
        unet_configs = {
            'dim': self.dim,
            'out_channels': self.out_channels,
            'resnet_block_groups': self.resnet_block_groups,
            'channels': self.channels,
            'dim_mults': self.dim_mults,
            'dtype': self.dtype,
            'self_condition': self.self_condition
        }
        x = Unet(**unet_configs)(x, time, x_self_cond)

        for _ in range(self.num_unets - 1):
            x = Unet(**unet_configs)(x, time, x_self_cond) + x
        return x


class UVit(nn.Module):
    dim: int = 384
    patch: int = 2
    out_channels: int = 3
    depth: int = 12
    # resnet_block_groups: int = 8,
    dtype: Any = jnp.bfloat16
    self_condition: bool = False

    @nn.compact
    def __call__(self, x, time, x_self_cond=None, *args, **kwargs):

        if x_self_cond is not None and self.self_condition:
            x = jnp.concatenate([x, x_self_cond], axis=3)
        elif self.self_condition:
            x = jnp.concatenate([x, jnp.zeros_like(x)], axis=3)
        print(x.shape)

        time_dim = self.dim * 4
        t = nn.Sequential([
            SinusoidalPosEmb(self.dim),
            nn.Dense(time_dim, dtype=self.dtype),
            nn.gelu,
            nn.Dense(time_dim, dtype=self.dtype)
        ])(time)

        x = nn.Conv(self.dim, (self.patch, self.patch), (self.patch, self.patch), padding="SAME", dtype=self.dtype)(x)
        r = x

        h = []

        for _ in range(self.depth // 2):
            x = Transformer(self.dim, self.dtype)(x)
            h.append(x)

        for _ in range(self.depth // 2):
            x = jnp.concatenate([x, h.pop()], axis=-1)
            x = nn.Dense(self.dim, dtype=self.dtype)(x)
            x = Transformer(self.dim, self.dtype)(x)

        x = einops.rearrange(x, 'b h w (c p1 p2)->b (h p1) (w p2) c', p1=self.patch, p2=self.patch)
        x = nn.Conv(self.out_channels, (3, 3), dtype="float32")(x)
        return x
