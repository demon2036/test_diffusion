import math
from functools import partial

import einops
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import *



class DownSample(nn.Module):
    dim: int
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        x = einops.rearrange(x, 'b  (h p1) (w p2) c -> b  h w (c p1 p2)', p1=2, p2=2)
        x = nn.Conv(self.dim, (1, 1),dtype=self.dtype)(x)
        return x


class Upsample(nn.Module):
    dim: int
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        b, h, w, c = x.shape
        x = jax.image.resize(x, shape=(b, h * 2, w * 2, c), method="nearest")
        x = nn.Conv(self.dim * 4, (3, 3), padding="SAME", dtype=self.dtype)(x)
        return x


# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x):
        half_dim = self.dim // 2
        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim, ) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis=-1)
        return emb



# class SinusoidalPosEmb(nn.Module):
#     dim: int
#     @nn.compact
#     def __call__(self, x, *args, **kwargs):
#         half_dim = self.dim // 2
#         emb = math.log(10000) / (half_dim - 1)
#         emb = jnp.exp(jnp.arange(half_dim, ) * -emb)
#         emb = x[:, None] * emb[None, :]
#         emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis=-1)
#         return emb


# building block modules

class Block(nn.Module):
    dim_out: int
    groups: int = 8
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, scale_shift=None):
        x = nn.Conv(self.dim_out, (3, 3), dtype=self.dtype, padding='SAME')(x)
        x = nn.GroupNorm(self.groups)(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = nn.silu(x)
        return x


class ResnetBlock(nn.Module):
    dim_out: int
    groups: int = 8
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, time_emb=None):
        _, _, _, c = x.shape
        scale_shift = None
        if time_emb is not None:
            time_emb = nn.silu(time_emb)
            time_emb = nn.Dense(self.dim_out * 2)(time_emb)
            time_emb = einops.rearrange(time_emb, 'b c -> b  1 1 c')
            scale_shift = jnp.split(time_emb,indices_or_sections=2,axis=3)

        h = Block(dim_out=self.dim_out,dtype=self.dtype)(x, scale_shift=scale_shift)

        h = Block(dim_out=self.dim_out,dtype=self.dtype)(h)

        if c != self.dim_out:
            x = nn.Conv(self.dim_out, (1, 1), dtype=self.dtype)(x)

        return h + x


class Unet(nn.Module):
    dim: int = 64
    out_channels:int =3
    resnet_block_groups: int = 8,
    channels: int = 3,
    dim_mults: Sequence = (1, 2, 4, 8)
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x,time,*args, **kwargs):
        time_dim = self.dim * 4
        t = nn.Sequential([
            SinusoidalPosEmb(self.dim),
            nn.Dense(time_dim,dtype=self.dtype),
            nn.gelu,
            nn.Dense(time_dim,dtype=self.dtype)
        ])(time)



        x = nn.Conv(self.dim, (7, 7), padding="SAME", dtype=self.dtype)(x)
        r = x

        h = []


        for i, dim_mul in enumerate(self.dim_mults):

            dim = self.dim * dim_mul

            x = ResnetBlock(dim,dtype=self.dtype)(x, t)
            h.append(x)
            x = ResnetBlock(dim,dtype=self.dtype)(x, t)
            h.append(x)
            if i != len(self.dim_mults) - 1:
                x = DownSample(dim,dtype=self.dtype)(x)
            else:
                x = nn.Conv(dim, (3, 3), dtype=self.dtype, padding="SAME")(x)

        x = ResnetBlock(dim, dtype=self.dtype)(x, t)
        # x = self.mid_attn(x) + x
        x = ResnetBlock(dim, dtype=self.dtype)(x, t)

        reversed_dim_mults = list(reversed(self.dim_mults))

        for i, dim_mul in enumerate(reversed_dim_mults):
            dim = self.dim * dim_mul

            x = jnp.concatenate([x, h.pop()], axis=3)
            x = ResnetBlock(dim,dtype=self.dtype)(x, t)
            x = jnp.concatenate([x, h.pop()], axis=3)
            x = ResnetBlock(dim,dtype=self.dtype)(x, t)

            if i != len(self.dim_mults) - 1:
                x = Upsample(dim,dtype=self.dtype)(x)
            else:
                x = nn.Conv(dim, (3, 3), dtype=self.dtype, padding="SAME")(x)





        x = jnp.concatenate([x, r], axis=3)

        x = ResnetBlock(dim, dtype=self.dtype)(x, t)
        x = nn.Conv(self.out_channels, (1, 1), dtype=self.dtype)(x)
        return x
