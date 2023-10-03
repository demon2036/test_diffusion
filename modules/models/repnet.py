import einops
import flax
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any


class RepBlock(nn.Module):
    dim: int
    dw_expand: int = 2
    dtype: Any = 'bfloat16'
    mlp_ratio: int = 4

    @nn.compact
    def __call__(self, x, time_emb=None, *args, **kwargs):
        y = x
        x = nn.GroupNorm(num_groups=8,dtype=self.dtype)(x)
        x = nn.silu(x)
        x = nn.Conv(self.dim, (1, 1),dtype=self.dtype)(x)
        # x_large = nn.Conv(self.dim, (15, 15), padding="SAME", feature_group_count=self.dim,dtype=self.dtype)(x)
        # x_small = nn.Conv(self.dim, (5, 5), padding="SAME", feature_group_count=self.dim,dtype=self.dtype)(x)
        x_large = nn.Conv(self.dim, (3, 3), padding="SAME",  dtype=self.dtype)(x)
        x_small = nn.Conv(self.dim, (1, 1), padding="SAME", dtype=self.dtype)(x)
        x = x + x_large + x_small
        x = nn.Conv(self.dim, (1, 1),dtype=self.dtype)(x)

        if y.shape[-1]!=self.dim:
            y = nn.Conv(self.dim, (1, 1),dtype=self.dtype)(y)

        y += x
        x = nn.GroupNorm(num_groups=8,dtype=self.dtype)(x)
        x = nn.Conv(self.dim * self.mlp_ratio, (1, 1),dtype=self.dtype)(x)
        x = nn.gelu(x)
        x = nn.Conv(self.dim, (1, 1),dtype=self.dtype)(x)
        return y + x


if __name__ == "__main__":
    pass
