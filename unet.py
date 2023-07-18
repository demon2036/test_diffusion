import math

import flax.linen as nn
from typing import *
import einops
import jax.random

from unet_block import EncoderBlock,DecoderBlock,MidBlock
import jax.numpy as jnp
import os
from diffusers import UNet2DModel
os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'

class SinusoidalPosEmb(nn.Module):
    dim:int
    @nn.compact
    def __call__(self, x,*args, **kwargs):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim, ) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis=-1)
        return emb





class Unet(nn.Module):
    out_channels: int
    layers_per_block: int
    block_out_channels: Sequence
    dtype: str
    scale: int = 1

    @nn.compact
    def __call__(self, x,time, *args, **kwargs):
        dim=self.block_out_channels[0]
        t_emb=SinusoidalPosEmb(dim)(time)

        t_emb=nn.Sequential([
            nn.Dense(dim * 4,dtype=self.dtype),
            nn.gelu,
            nn.Dense(dim * 4,dtype=self.dtype)
        ])(t_emb)


        block_out_channels = self.block_out_channels
        # x = einops.rearrange(x, 'b  (h p1) (w p2) c->b h w (c p1 p2)', p1=self.scale, p2=self.scale)
        x=nn.Conv(block_out_channels[0],(3,3),padding="SAME",dtype=self.dtype)(x)
        temp = []

        for i, channesls in enumerate(block_out_channels):
            is_final = (i == len(block_out_channels) - 1)
            x, skip_block = EncoderBlock(channesls, self.layers_per_block, dtype=self.dtype,
                                         down=True if not is_final else False)(x,t_emb)

            temp.append(skip_block)

        prev = MidBlock(block_out_channels[-1], self.layers_per_block, dtype=self.dtype)(x,t_emb)

        skip_blocks = list(reversed(temp))
        block_out_channels = list(reversed(block_out_channels))

        for i, (skip_block, channels) in enumerate(zip(skip_blocks, block_out_channels)):
            is_final = (i == len(block_out_channels) - 1)
            x = DecoderBlock(channels, self.layers_per_block, dtype=self.dtype, up=True if not is_final else False)(prev, skip_block,t_emb)
            prev = x
        x = nn.GroupNorm()(x)
        x = nn.silu(x)
        x = nn.Conv(self.out_channels * self.scale ** 2, (3, 3), padding='SAME', dtype='float32')(x)
        # x = einops.rearrange(x, 'b h w (c p1 p2)->b  (h p1) (w p2) c', p1=self.scale, p2=self.scale)
        return x

if __name__=="__main__":
    key = jax.random.PRNGKey(seed=43)
    dim = 512
    s=SinusoidalPosEmb(dim)

    variable=s.init(key,jnp.ones((1,)))

    t=s.apply({},jnp.ones((1,))  )
    print(t.shape)
    print(t)
