import einops
import flax
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any
import os

os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'


class SimpleGate(nn.Module):
    @nn.compact
    def __call__(self, x):
        x1, x2 = jnp.split(x, 2, 3)
        return x1 * x2


class NAFBlock(nn.Module):
    dim: int
    dw_expand: int = 2
    dtype: Any = 'bfloat16'

    @nn.compact
    def __call__(self, x, time_emb=None, *args, **kwargs):

        hidden = nn.LayerNorm(dtype=self.dtype)(x)
        if time_emb is not None:
            time_emb = nn.silu(time_emb)
            time_emb = nn.Dense(hidden.shape[-1] * 2, dtype=self.dtype)(time_emb)
            time_emb = einops.rearrange(time_emb, 'b c -> b  1 1 c')
            scale_att, shift_att = jnp.split(time_emb, indices_or_sections=2, axis=3)
            hidden = hidden * (scale_att + 1) + shift_att

        hidden = nn.Conv(self.dim * self.dw_expand, (1, 1), dtype=self.dtype)(hidden)
        hidden = nn.Conv(self.dim * self.dw_expand, (3, 3), feature_group_count=self.dim * self.dw_expand,
                         padding="SAME", dtype=self.dtype)(hidden)

        hidden = SimpleGate()(hidden)
        shape = hidden.shape
        pool_size = (shape[1], shape[2])
        pool = nn.avg_pool(hidden, pool_size, pool_size)
        attn = nn.Conv(self.dim, (1, 1), dtype=self.dtype)(pool)

        hidden = hidden * attn

        hidden = nn.Conv(self.dim, (1, 1), dtype=self.dtype)(hidden)

        if self.dim != x.shape[-1]:
            x = nn.Conv(self.dim, (1, 1), dtype=self.dtype)(x)

        y = x + hidden

        hidden = nn.LayerNorm(dtype=self.dtype)(x)
        if time_emb is not None:
            time_emb = nn.silu(time_emb)
            time_emb = nn.Dense(self.dim * 2, dtype=self.dtype)(time_emb)
            #time_emb = einops.rearrange(time_emb, 'b c -> b  1 1 c')
            scale_ffn, shift_ffn = jnp.split(time_emb, indices_or_sections=2, axis=3)
            hidden = hidden * (scale_ffn + 1) + shift_ffn

        hidden = nn.Conv(self.dim * self.dw_expand, (1, 1), dtype=self.dtype)(hidden)
        hidden = SimpleGate()(hidden)
        hidden = nn.Conv(self.dim, (1, 1), dtype=self.dtype)(hidden)

        return y + hidden


if __name__ == "__main__":
    pass
