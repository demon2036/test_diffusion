import einops
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any

from modules.models.embedding import SinusoidalPosEmb


class MLP(nn.Module):
    dim: int
    dtype: Any = 'bfloat16'

    @nn.compact
    def __call__(self, x, time_emb=None, *args, **kwargs):
        x = nn.Dense(self.dim, dtype=self.dtype)(x)

        if time_emb is not None:
            time_emb = nn.silu(time_emb)
            time_emb = nn.Dense(self.dim * 2, dtype=self.dtype)(time_emb)
            scale_shift = jnp.split(time_emb, indices_or_sections=2, axis=-1)
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = nn.LayerNorm(dtype=self.dtype)(x)
        x = nn.silu(x)
        return x


class LatentNet(nn.Module):
    num_layers: int = 10
    dim: int = 2048
    out_dim: int = 512
    dtype: Any = 'bfloat16'

    @nn.compact
    def __call__(self, x, time, *args, **kwargs):

        time_dim = self.dim * 1
        t = nn.Sequential([
            SinusoidalPosEmb(self.dim),
            nn.Dense(time_dim, dtype=self.dtype),
            nn.gelu,
            nn.Dense(time_dim, dtype=self.dtype)
        ])(time)

        h = x
        for i in range(self.num_layers):
            if i != 0:
                h = jnp.concatenate([h, x], axis=-1)
            h = MLP(self.dim, self.dtype)(h, t)

        # x = nn.LayerNorm(dtype=self.dtype)(x)
        # x = nn.silu(x)
        x = nn.Dense(self.out_dim, dtype='float32')(h)
        return x


class LatentNet2D(nn.Module):
    num_layers: int = 10
    dim: int = 2048
    out_dim: int = 4096
    dtype: Any = 'bfloat16'

    @nn.compact
    def __call__(self, x, time, *args, **kwargs):
        print(f'x.shape:{x.shape}')
        b, h, w, c = x.shape
        x = einops.rearrange(x, 'b h w c->b (h w c)')

        time_dim = self.dim * 1
        t = nn.Sequential([
            SinusoidalPosEmb(self.dim),
            nn.Dense(time_dim, dtype=self.dtype),
            nn.gelu,
            nn.Dense(time_dim, dtype=self.dtype)
        ])(time)

        hidden = x
        for i in range(self.num_layers):
            if i != 0:
                hidden = jnp.concatenate([hidden, x], axis=-1)
            hidden = MLP(self.dim, self.dtype)(hidden, t)

        # x = nn.LayerNorm(dtype=self.dtype)(x)
        # x = nn.silu(x)
        x = nn.Dense(self.out_dim, dtype='float32')(jnp.concatenate([hidden, x], axis=-1))
        x = einops.rearrange(x, 'b (h w c)->b h w c', h=h, w=w, c=c)

        return x
