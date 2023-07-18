from jax._src.nn.initializers import constant
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import *
from functools import partial
from resnet import ResBlock,DownSample,Upsample


class EncoderBlock(nn.Module):
    features: int
    layers_per_block: int
    dtype: str
    down: bool

    @nn.compact
    def __call__(self, x,t_emb, *args, **kwargs):
        skip_blocks = []
        for _ in range(self.layers_per_block):
            x = ResBlock(features=self.features, dtype=self.dtype)(x,t_emb)
            skip_blocks.append(x)
        if self.down:
            x = DownSample(self.features, dtype=self.dtype)(x)
        return x, skip_blocks



class MidBlock(nn.Module):
    features: int
    layers_per_block: int
    dtype: str

    @nn.compact
    def __call__(self, x,t_emb, *args, **kwargs):
        for _ in range(self.layers_per_block):
            x = ResBlock(self.features, dtype=self.dtype)(x,t_emb)

        return x


class DecoderBlock(nn.Module):
    features: int
    layers_per_block: int
    dtype: str
    up: bool

    @nn.compact
    def __call__(self, x, skip_blocks: list,t_emb, *args, **kwargs):
        for _ in range(self.layers_per_block):
            prev_block = skip_blocks.pop()
            x = jnp.concatenate([x, prev_block], axis=3)
            x = ResBlock(self.features, dtype=self.dtype)(x,t_emb)
        if self.up:
            x = Upsample(self.features, dtype=self.dtype)(x)
        return x