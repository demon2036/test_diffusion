import einops
from jax._src.nn.initializers import constant
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import *
from functools import partial


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
    kernel_size: int = 3
    dtype: str = 'bfloat16'

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        _, _, _, c = x.shape
        x = nn.Conv(self.features, (self.kernel_size, self.kernel_size), padding="SAME", dtype=self.dtype,
                    feature_group_count=c)(x)
        x = nn.Conv(self.features, (1, 1), dtype=self.dtype)(x)
        return x


class Block(nn.Module):
    dim_out: int
    groups: int = 8
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, scale_shift=None):
        x = nn.Conv(self.dim_out, (3, 3), dtype=self.dtype, padding='SAME')(x)
        x = nn.GroupNorm(self.groups, dtype=self.dtype, )(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = nn.silu(x)
        return x


class ResBlock(nn.Module):
    dim_out: int
    groups: int = 32
    dtype: Any = jnp.bfloat16
    @nn.compact
    def __call__(self, x, time_emb=None):
        _, _, _, c = x.shape
        scale_shift = None
        if time_emb is not None:
            time_emb = nn.silu(time_emb)
            time_emb = nn.Dense(self.dim_out * 2, dtype=self.dtype)(time_emb)
            time_emb = einops.rearrange(time_emb, 'b c -> b  1 1 c')
            scale_shift = jnp.split(time_emb, indices_or_sections=2, axis=3)

        h = Block(dim_out=self.dim_out, dtype=self.dtype)(x, scale_shift=scale_shift)
        h = Block(dim_out=self.dim_out, dtype=self.dtype)(h)
        if c != self.dim_out:
            x = nn.Conv(self.dim_out, (1, 1), dtype=self.dtype)(x)

        return h + x


class EfficientBlock(nn.Module):
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
        x = nn.GroupNorm(8)(x)

        x = einops.rearrange(x, 'b h w (c f)->b h w (f c)', f=self.factor)

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


class EfficientBlock2(nn.Module):
    features: int
    dtype: str
    factor: int = 4

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        b, h, w, c = x.shape
        partial_channel = c // self.factor
        conv = partial(nn.Conv, padding='SAME', dtype=self.dtype, feature_group_count=partial_channel)
        # x = einops.rearrange(x, 'b h w (c f)->b h w (f c)', f=self.factor)
        out_put = []
        count = 3

        for i in range(0, self.factor):
            first_part = x[:, :, :, i * partial_channel: (i + 1) * partial_channel]
            first_part = conv(partial_channel, (count, count))(first_part)
            count += 2
            out_put.append(first_part)
        x = jnp.concatenate(out_put, axis=3)

        x = nn.GroupNorm(8)(x)
        x = nn.silu(x)
        x = conv(self.features, (1, 1), feature_group_count=1)(x)
        return x


class DownSample(nn.Module):
    dim: int
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        x = einops.rearrange(x, 'b  (h p1) (w p2) c -> b  h w (c p1 p2)', p1=2, p2=2)
        x = nn.Conv(self.dim, (1, 1), dtype=self.dtype)(x)
        return x


class UpSample(nn.Module):
    dim: int
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        b, h, w, c = x.shape
        x = jax.image.resize(x, shape=(b, h * 2, w * 2, c), method="nearest")
        x = nn.Conv(self.dim, (3, 3), padding="SAME", dtype=self.dtype)(x)
        return x
