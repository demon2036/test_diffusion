import einops
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from typing import *
from functools import partial
from einops import rearrange
from jax._src.nn.initializers import constant

from diffusers import DDIMScheduler


class WeightStandardizedConv(nn.Module):
    """
    apply weight standardization  https://arxiv.org/abs/1903.10520
    """
    features: int
    kernel_size: Sequence[int] = 3
    strides: Union[None, int, Sequence[int]] = 1
    padding: Any = "SAME"
    kernel_init: Any = nn.initializers.lecun_normal()
    kernel_dilation: Sequence[int] = 1
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    feature_group_count: int = 1

    @nn.compact
    def __call__(self, x):
        """
        Applies a weight standardized convolution to the inputs.

        Args:
          inputs: input data with dimensions (batch, spatial_dims..., features).

        Returns:
          The convolved data.
        """
        x = x.astype(self.dtype)

        conv = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            dtype=self.dtype,
            kernel_dilation=self.kernel_dilation,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            feature_group_count=self.feature_group_count,
            parent=None, precision="highest")

        kernel_init = lambda rng, x: conv.init(rng, x)['params']['kernel']
        bias_init = lambda rng, x: conv.init(rng, x)['params']['bias']

        # standardize kernel
        kernel = self.param('kernel', kernel_init, x)
        kernel_norm = jnp.linalg.norm(kernel)
        # []
        norm = self.param('norm', constant(kernel_norm), [])
        # reduce over dim_out
        # redux = tuple(range(kernel.ndim - 1))
        # mean = jnp.mean(kernel, axis=redux, dtype=self.dtype, keepdims=True)
        # var = jnp.var(kernel, axis=redux, dtype=self.dtype, keepdims=True)
        standardized_kernel = norm * kernel / kernel_norm

        bias = self.param('bias', bias_init, x)

        return (conv.apply({'params': {'kernel': standardized_kernel, 'bias': bias}}, x))


class DepthWiseConv(nn.Module):
    features: int
    kernel_size: int
    dtype: str

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        x = nn.Conv(self.features, (self.kernel_size, self.kernel_size), padding="SAME", dtype=self.dtype)(x)
        x = nn.Conv(self.features, (1, 1), dtype=self.dtype)(x)
        return x


class DownSample(nn.Module):
    features: int
    dtype: str

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        x = nn.Conv(self.features, (3, 3), (2, 2), padding="SAME", dtype=self.dtype)(x)
        return x


class Upsample(nn.Module):
    features: int
    dtype: str

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        x = nn.Conv(self.features * 4, (3, 3), padding="SAME", dtype=self.dtype)(x)
        x = rearrange(x, ' b h w (c p1 p2)->b (h p1) (w p2) c', p1=2, p2=2)
        return x


class ResBlock(nn.Module):
    features: int
    dtype: str
    factor: int = 4

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        project_channels = self.features

        partial_channel = project_channels // self.factor
        conv = partial(nn.Conv, padding='SAME', dtype=self.dtype)
        x = conv(project_channels, (1, 1))(x)
        y = x
        x = nn.GroupNorm()(x)
        out_put = []
        first_part = x[:, :, :, :partial_channel]
        for i in range(1, self.factor):
            first_part = jnp.concatenate([first_part,
                                          x[:, :, :, i * partial_channel: (i + 1) * partial_channel]],
                                         axis=3)
            first_part = conv(partial_channel, (1, 1))(first_part)
            first_part = nn.hard_swish(first_part)
            first_part = conv(partial_channel, (3, 3))(first_part)
            out_put.append(first_part)
        x = jnp.concatenate(out_put, axis=3)
        x = conv(self.features, (1, 1))(x) + y
        return x


"""
class ResBlock(nn.Module):
    features: int
    dtype: str

    # groups:int
    @nn.compact
    def __call__(self, x, *args, **kwargs):
        # use or not use short cut
        _, _, _, c = x.shape
        if c == self.features:
            y = x
        else:
            y = nn.Conv(self.features, (1, 1), dtype=self.dtype)(x)
        x = DepthWiseConv(self.features, 5, dtype=self.dtype)(x)
        # x = nn.LayerNorm()(x)
        x = nn.GroupNorm()(x)
        x = nn.Conv(self.features * 4, (1, 1), dtype=self.dtype)(x)
        x = nn.silu(x)
        # x = nn.gelu(x)
        x = nn.Conv(self.features, (1, 1), dtype=self.dtype)(x)
        return x + y
"""


class EncoderBlock(nn.Module):
    features: int
    layers_per_block: int
    dtype: str
    down: bool

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        skip_blocks = []
        for _ in range(self.layers_per_block):
            x = ResBlock(features=self.features, dtype=self.dtype)(x)
            skip_blocks.append(x)
        if self.down:
            x = DownSample(self.features, dtype=self.dtype)(x)
        return x, skip_blocks


class MidBlock(nn.Module):
    features: int
    layers_per_block: int
    dtype: str

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        for _ in range(self.layers_per_block):
            x = ResBlock(self.features, dtype=self.dtype)(x)

        return x


class DecoderBlock(nn.Module):
    features: int
    layers_per_block: int
    dtype: str
    up: bool

    @nn.compact
    def __call__(self, x, skip_blocks: list, *args, **kwargs):
        for _ in range(self.layers_per_block):
            prev_block = skip_blocks.pop()
            x = jnp.concatenate([x, prev_block], axis=3)
            x = ResBlock(self.features, dtype=self.dtype)(x)
        if self.up:
            x = Upsample(self.features, dtype=self.dtype)(x)
        return x


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
