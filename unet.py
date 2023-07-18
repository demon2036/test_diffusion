import einops
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from typing import *
from functools import partial
from einops import rearrange

from unet_block import EncoderBlock,DecoderBlock,MidBlock






class Unet(nn.Module):
    out_channels: int
    layers_per_block: int
    block_out_channels: Sequence
    dtype: str
    scale: int = 1

    @nn.compact
    def __call__(self, x, *args, **kwargs):

        block_out_channels = self.block_out_channels

        # x = einops.rearrange(x, 'b  (h p1) (w p2) c->b h w (c p1 p2)', p1=self.scale, p2=self.scale)
        x=nn.Conv(block_out_channels[0],(3,3),padding="SAME",dtype=self.dtype)(x)
        temp = []

        for i, channesls in enumerate(block_out_channels):
            is_final = (i == len(block_out_channels) - 1)

            x, skip_block = EncoderBlock(channesls, self.layers_per_block, dtype=self.dtype,
                                         down=True if not is_final else False)(x)

            temp.append(skip_block)

        prev = MidBlock(block_out_channels[-1], self.layers_per_block, dtype=self.dtype)(x)

        skip_blocks = list(reversed(temp))
        block_out_channels = list(reversed(block_out_channels))

        for i, (skip_block, channels) in enumerate(zip(skip_blocks, block_out_channels)):
            is_final = (i == len(block_out_channels) - 1)
            x = DecoderBlock(channels, self.layers_per_block, dtype=self.dtype, up=True if not is_final else False)(prev, skip_block)
            prev = x
        x = nn.GroupNorm()(x)
        x = nn.silu(x)
        x = nn.Conv(self.out_channels * self.scale ** 2, (3, 3), padding='SAME', dtype='float32')(x)
        # x = einops.rearrange(x, 'b h w (c p1 p2)->b  (h p1) (w p2) c', p1=self.scale, p2=self.scale)
        return x
