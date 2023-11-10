import math
from functools import partial
import einops
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import *
from einops.layers.flax import Rearrange

from modules.models.attention import Attention
from modules.models.nafnet import NAFBlock
from modules.models.autoencoder import Encoder, AutoEncoderKL, AutoEncoder, Encoder2DLatent
from modules.models.transformer import Transformer
from modules.models.embedding import SinusoidalPosEmb
from modules.models.resnet import ResBlock, DownSample, UpSample, EfficientBlock


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


class GlobalAveragePool(nn.Module):
    @nn.compact
    def __call__(self, x, *args, **kwargs):
        x = nn.avg_pool(x, window_shape=(x.shape[1], x.shape[2]), strides=(1, 1))
        return x


class Encoder1DLatent(nn.Module):
    latent: int

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        b, h, w, c = x.shape
        x = nn.GroupNorm(min(c, 32))(x)
        x = nn.silu(x)
        x = nn.Conv(self.latent, (1, 1))(x)
        # Global Average Pooling
        x = nn.avg_pool(x, window_shape=(x.shape[1], x.shape[2]), strides=(1, 1))
        x = nn.Conv(self.latent, (1, 1))(x)
        x = einops.rearrange(x, 'b h w c->b (h w c)')
        return x


class Unet(nn.Module):
    dim: int = 64
    dim_mults: Sequence = (1, 2, 4, 4)
    num_res_blocks: Any = 2
    num_middle_blocks: Any = 2
    out_channels: int = 3
    resnet_block_groups: int = 8,
    channels: int = 3,
    dtype: Any = jnp.bfloat16
    self_condition: bool = False
    use_encoder: bool = False
    encoder_type: str = '2D'
    res_type: Any = 'default'
    patch_size: int = 1
    residual: bool = False
    n: int = 8
    time_embedding: bool = True

    @nn.compact
    def __call__(self, x, time, x_self_cond=None, sr_factors=None, z_rng=None, *args, **kwargs):
        b, h, w, c = x.shape
        y = x
        if type(self.num_res_blocks) == int:
            num_res_blocks = (self.num_res_blocks,) * len(self.dim_mults)
        else:
            assert len(self.num_res_blocks) == len(self.dim_mults)
            num_res_blocks = self.num_res_blocks

        if self.res_type == 'default':
            res_block = ResBlock
        elif self.res_type == "NAF":
            res_block = NAFBlock
        elif self.res_type == "efficient":
            res_block = EfficientBlock
        else:
            raise NotImplemented()

        cond_emb = None
        if self.use_encoder:
            latent = x_self_cond
            # assert self.encoder_type in ['1D', '2D', 'Both']
            print(f'latent shape:{latent.shape}')
            if self.encoder_type == '1D':
                cond_emb = latent
                x_self_cond = None
            elif self.encoder_type == '2D':
                cond_emb = None

                x_self_cond = Encoder2DLatent(shape=x.shape, n=self.n)(latent)

                if z_rng is not None:
                    print(f'z_rng:{z_rng}')
                    z_rng, p_key = jax.random.split(z_rng)
                    p = jax.random.uniform(p_key, (b,))
                    mask = jax.random.bernoulli(z_rng, p, shape=(1, h, w, b)).reshape(b, h, w, 1)
                    x_self_cond = x_self_cond * mask
                else:
                    print(f'z_rng:{z_rng}')

            elif self.encoder_type == '2D_as_1D':
                cond_emb = Encoder1DLatent(2048)(x)
                x_self_cond = None
            elif self.encoder_type == 'Both':
                cond_emb = nn.Sequential([
                    nn.GroupNorm(num_groups=min(8, latent.shape[-1])),
                    nn.silu,
                    nn.Conv(self.dim * 16, (1, 1)),
                    GlobalAveragePool(),
                    Rearrange('b h w c->b (h w c)'),
                    nn.Dense(self.dim * 16)
                ])(latent)
                x_self_cond = Encoder2DLatent(shape=x.shape)(latent)
            else:
                raise NotImplemented()

        if x_self_cond is not None:
            x = jnp.concatenate([x, x_self_cond], axis=3)

        time_dim = self.dim * 4

        if self.time_embedding:
            t = nn.Sequential([
                SinusoidalPosEmb(self.dim),
                nn.Dense(time_dim, dtype=self.dtype),
                nn.gelu,
                nn.Dense(time_dim, dtype=self.dtype)
            ])(time)
        else:
            t = None

        if sr_factors is not None:
            t += nn.Sequential([
                SinusoidalPosEmb(self.dim),
                nn.Dense(time_dim, dtype=self.dtype),
                nn.gelu,
                nn.Dense(time_dim, dtype=self.dtype)
            ])(sr_factors)

        x = einops.rearrange(x, 'b (h p1) (w p2) c->b h w (c p1 p2)', p1=self.patch_size, p2=self.patch_size)

        x = nn.Conv(self.dim, (3, 3), (1, 1), padding="SAME",
                    dtype=self.dtype)(x)
        r = x

        h = [x]

        for i, (dim_mul, num_res_block) in enumerate(zip(self.dim_mults, num_res_blocks)):
            dim = self.dim * dim_mul
            for _ in range(num_res_block):
                x = res_block(dim, dtype=self.dtype)(x, t, cond_emb)
                # x=Attention(dim=dim,dtype=self.dtype)(x)+x
                h.append(x)

            if i != len(self.dim_mults) - 1:
                x = DownSample(dim, dtype=self.dtype)(x)
                h.append(x)
            # else:
            #     x = nn.Conv(dim, (3, 3), dtype=self.dtype, padding="SAME")(x)

        for _ in range(self.num_middle_blocks):
            x = res_block(dim, dtype=self.dtype)(x, t, cond_emb)

        reversed_dim_mults = list(reversed(self.dim_mults))
        reversed_num_res_blocks = list(reversed(num_res_blocks))

        for i, (dim_mul, num_res_block) in enumerate(zip(reversed_dim_mults, reversed_num_res_blocks)):
            dim = self.dim * dim_mul
            for _ in range(num_res_block + 1):
                x = jnp.concatenate([x, h.pop()], axis=3)
                x = res_block(dim, dtype=self.dtype)(x, t, cond_emb)
                # x = Attention(dim=dim, dtype=self.dtype)(x) + x

            if i != len(self.dim_mults) - 1:
                x = UpSample(dim, dtype=self.dtype)(x)
            # else:
            #     x = nn.Conv(dim, (3, 3), dtype=self.dtype, padding="SAME")(x)

        # x = jnp.concatenate([x, r], axis=3)
        # x = res_block(dim, dtype=self.dtype)(x, t)
        x = nn.GroupNorm()(x)
        x = nn.Conv(self.out_channels * self.patch_size ** 2, (3, 3), dtype="float32")(x)
        # x = einops.rearrange(x, 'b h w (c p1 p2)->b (h p1) (w p2) c', p1=self.patch_size, p2=self.patch_size)

        if self.residual:
            x = x + y

        return x


class UnetTest(nn.Module):
    dim: int = 64
    dim_mults: Sequence = (1, 2, 4, 4)
    kernel_size: Any = 3
    num_res_blocks: Any = 2
    num_middle_blocks: Any = 2
    out_channels: int = 3
    resnet_block_groups: int = 8,
    channels: int = 3,
    dtype: Any = jnp.bfloat16
    self_condition: bool = False
    use_encoder: bool = False
    encoder_type: str = '2D'
    res_type: Any = 'default'
    patch_size: int = 1
    residual: bool = False
    n: int = 8
    skip_connection: str = 'add'

    @nn.compact
    def __call__(self, x, time, x_self_cond=None, z_rng=None, *args, **kwargs):

        y = x

        if type(self.num_res_blocks) == int:
            num_res_blocks = (self.num_res_blocks,) * len(self.dim_mults)
        else:
            assert len(self.num_res_blocks) == len(self.dim_mults)
            num_res_blocks = self.num_res_blocks

        if type(self.kernel_size) == int:
            kernel_size = (self.kernel_size,) * len(self.dim_mults)
        else:
            assert len(self.kernel_size) == len(self.dim_mults)
            kernel_size = self.kernel_size

        if self.res_type == 'default':
            res_block = ResBlock
        elif self.res_type == "NAF":
            res_block = NAFBlock
        elif self.res_type == "efficient":
            res_block = EfficientBlock
        else:
            raise NotImplemented()

        cond_emb = None
        if self.use_encoder:
            latent = x_self_cond
            assert self.encoder_type in ['1D', '2D', 'Both', '2D_as_1D']
            print(f'latent shape:{latent.shape}')
            if self.encoder_type == '1D':
                cond_emb = latent
                x_self_cond = None
            elif self.encoder_type == '2D':
                cond_emb = None
                x_self_cond = Encoder2DLatent(shape=x.shape, n=self.n)(latent)
            elif self.encoder_type == 'Both':
                cond_emb = nn.Sequential([
                    nn.GroupNorm(num_groups=min(8, latent.shape[-1])),
                    nn.silu,
                    nn.Conv(self.dim * 16, (1, 1)),
                    GlobalAveragePool(),
                    Rearrange('b h w c->b (h w c)'),
                    nn.Dense(self.dim * 16)
                ])(latent)
                x_self_cond = Encoder2DLatent(shape=x.shape)(latent)
            elif self.encoder_type == '2D_as_1D':
                cond_emb = None
                x_self_cond = Encoder1DLatent(shape=x.shape, n=self.n)(latent)

        if x_self_cond is not None:
            x = jnp.concatenate([x, x_self_cond], axis=3)

        time_dim = self.dim * 4
        t = nn.Sequential([
            SinusoidalPosEmb(self.dim),
            nn.Dense(time_dim, dtype=self.dtype),
            nn.gelu,
            nn.Dense(time_dim, dtype=self.dtype)
        ])(time)

        x = einops.rearrange(x, 'b (h p1) (w p2) c->b h w (c p1 p2)', p1=self.patch_size, p2=self.patch_size)

        x = nn.Conv(self.dim, (3, 3), padding="SAME",
                    dtype=self.dtype)(x)
        r = x

        h = [x]

        for i, (dim_mul, num_res_block, block_kernel_size) in enumerate(
                zip(self.dim_mults, num_res_blocks, kernel_size)):
            dim = self.dim * dim_mul
            for _ in range(num_res_block):
                x = res_block(dim, dtype=self.dtype, kernel_size=block_kernel_size)(x, t, cond_emb)
                x = Attention(dim)(x) + x
                h.append(x)

            if i != len(self.dim_mults) - 1:
                x = DownSample(self.dim * self.dim_mults[i + 1], dtype=self.dtype)(x)
                h.append(x)
            # else:
            #     x = nn.Conv(dim, (3, 3), dtype=self.dtype, padding="SAME")(x)

        for _ in range(self.num_middle_blocks):
            x = res_block(dim, dtype=self.dtype)(x, t, cond_emb)
            x = Attention(dim)(x) + x

        reversed_kernel_size = list(reversed(kernel_size))
        reversed_dim_mults = list(reversed(self.dim_mults))
        reversed_num_res_blocks = list(reversed(num_res_blocks))

        for i, (dim_mul, num_res_block, block_kernel_size) in enumerate(
                zip(reversed_dim_mults, reversed_num_res_blocks, reversed_kernel_size)):
            dim = self.dim * dim_mul
            for _ in range(num_res_block + 1):
                if self.skip_connection == 'concat':
                    x = jnp.concatenate([x, h.pop()], axis=3)
                elif self.skip_connection == 'add':
                    x = x + h.pop() * (2 ** -0.5)
                else:
                    raise NotImplemented()

                x = res_block(dim, dtype=self.dtype, kernel_size=block_kernel_size)(x, t, cond_emb)
                x = Attention(dim)(x) + x

            if i != len(self.dim_mults) - 1:
                x = UpSample(self.dim * reversed_dim_mults[i + 1], dtype=self.dtype)(x)
            # else:
            #     x = nn.Conv(dim, (3, 3), dtype=self.dtype, padding="SAME")(x)

        # x = jnp.concatenate([x, r], axis=3)
        # x = res_block(dim, dtype=self.dtype)(x, t)
        x = nn.GroupNorm()(x)
        x = nn.silu(x)
        x = nn.Conv(self.out_channels * self.patch_size ** 2, (3, 3), dtype="float32")(x)
        x = einops.rearrange(x, 'b h w (c p1 p2)->b (h p1) (w p2) c', p1=self.patch_size, p2=self.patch_size)

        if self.residual:
            x = x + y

        return x
